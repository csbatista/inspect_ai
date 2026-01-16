import functools
import os
import time
from datetime import datetime, timezone
from typing import Any, TypeAlias

import pydantic
from google.genai import Client
from google.genai.types import (
    CreateBatchJobConfig,
    GenerateContentResponse,
    HttpOptions,
    JobError,
    JobState,
)
from typing_extensions import override

from inspect_ai.model._generate_config import BatchConfig
from inspect_ai.model._retry import ModelRetryConfig

from .util.batch import Batch, BatchCheckResult, BatchRequest
from .util.file_batcher import FileBatcher
from .util.hooks import HttpxHooks

# Just the result URI
CompletedBatchInfo: TypeAlias = str


class GoogleVertexBatcher(FileBatcher[GenerateContentResponse, CompletedBatchInfo]):
    """Vertex AI batch processor using GCS for file storage.

    This implementation uses Google Cloud Storage (GCS) for uploading input files
    and downloading result files, which is the recommended approach for Vertex AI
    batch processing.
    """

    def __init__(
        self,
        client: Client,
        config: BatchConfig,
        retry_config: ModelRetryConfig,
        model_name: str,
        project: str | None = None,
        location: str | None = None,
        gcs_bucket: str | None = None,
    ):
        super().__init__(
            config=config,
            retry_config=retry_config,
            max_batch_request_count=50000,
            max_batch_size_mb=2000,  # 2GB file size limit
        )
        self._client = client
        self._model_name = model_name

        # Get project and location from args or environment
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")

        # GCS bucket for batch files (use environment variable or default)
        # Format: gs://bucket-name or just bucket-name
        self._gcs_bucket = (
            gcs_bucket
            or os.environ.get("GOOGLE_VERTEX_BATCH_BUCKET")
            or os.environ.get("VERTEX_BATCH_BUCKET")
        )

        if not self._project:
            raise ValueError(
                "Project must be specified via constructor argument, "
                "GOOGLE_CLOUD_PROJECT environment variable, or model_args"
            )
        if not self._location:
            raise ValueError(
                "Location must be specified via constructor argument, "
                "GOOGLE_CLOUD_LOCATION environment variable, or model_args"
            )

        # Initialize GCS client lazily
        self._gcs_client: Any = None

    def _get_gcs_client(self) -> Any:
        """Lazy initialization of GCS client."""
        if self._gcs_client is None:
            try:
                from google.cloud import storage

                self._gcs_client = storage.Client(project=self._project)
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required for Vertex AI batch processing. "
                    "Install it with: pip install google-cloud-storage"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize GCS client for project '{self._project}'. "
                    f"Ensure you have proper authentication configured (e.g., via gcloud auth or "
                    f"GOOGLE_APPLICATION_CREDENTIALS environment variable). Error: {e}"
                ) from e
        return self._gcs_client

    def _get_gcs_bucket_name(self) -> str:
        """Get the GCS bucket name, removing gs:// prefix if present."""
        if not self._gcs_bucket:
            # Default bucket name pattern
            return f"{self._project}-vertex-ai-batch"

        bucket = self._gcs_bucket
        if bucket.startswith("gs://"):
            bucket = bucket[5:]
        # Remove trailing slashes
        return bucket.rstrip("/")

    def _get_gcs_path(self, filename: str) -> str:
        """Generate a GCS path for a batch file."""
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        return f"inspect-ai-batches/{timestamp}/{filename}"

    @override
    def _jsonl_line_for_request(
        self, request: BatchRequest[GenerateContentResponse], custom_id: str
    ) -> dict[str, pydantic.JsonValue]:
        """Format request for Vertex AI batch processing.

        Vertex AI batch uses a different format than Google AI:
        - Uses "systemInstruction" (camelCase) with role="system" and parts array
        - Uses "generationConfig" (camelCase) instead of individual config fields
        - Doesn't support safety_settings in batch mode
        - Doesn't support candidate_count
        """
        # Fields not supported by Vertex AI batch API or need transformation
        vertex_incompatible_fields = {
            "http_options",  # Internal to client, not part of API
            "candidate_count",  # Not supported by Vertex AI batch
            "system_instruction",  # Will be transformed to systemInstruction
            "safety_settings",  # Not supported in batch mode
        }

        # Also remove individual generation config fields that should be in generationConfig
        generation_config_fields = {
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "stop_sequences",
            "presence_penalty",
            "frequency_penalty",
            "response_logprobs",
            "logprobs",
            "response_mime_type",
            "response_schema",
            "thinking_config",
        }

        # Build the request
        filtered_request = {}
        generation_config = {}

        for k, v in request.request.items():
            if k in vertex_incompatible_fields:
                continue
            elif k in generation_config_fields:
                # Collect generation config fields
                generation_config[k] = v
            else:
                filtered_request[k] = v

        # Add generationConfig if we collected any fields
        if generation_config:
            filtered_request["generationConfig"] = generation_config

        # Handle system_instruction transformation
        # Vertex AI batch expects: systemInstruction (camelCase) with role="system"
        if "system_instruction" in request.request:
            sys_instruction = request.request["system_instruction"]
            if sys_instruction:
                # Convert to Vertex AI format
                filtered_request["systemInstruction"] = {
                    "role": "system",
                    "parts": sys_instruction
                    if isinstance(sys_instruction, list)
                    else [sys_instruction],
                }

        return {
            "custom_id": custom_id,
            "request": filtered_request,
        }

    @override
    async def _upload_batch_file(
        self, temp_file: Any, extra_headers: dict[str, str]
    ) -> str:
        """Upload batch file to GCS and return the GCS URI."""
        import logging

        import anyio

        logger = logging.getLogger(__name__)

        try:
            gcs_client = self._get_gcs_client()
            bucket_name = self._get_gcs_bucket_name()

            logger.info(f"Uploading batch file to GCS bucket: {bucket_name}")

            # Generate unique filename
            filename = f"batch_requests_{int(time.time())}.jsonl"
            gcs_path = self._get_gcs_path(filename)
            gcs_uri = f"gs://{bucket_name}/{gcs_path}"

            # Get bucket and create blob
            bucket = gcs_client.bucket(bucket_name)

            # Check if bucket exists
            try:
                bucket_exists = await anyio.to_thread.run_sync(
                    bucket.exists, cancellable=True
                )
                if not bucket_exists:
                    raise RuntimeError(
                        f"GCS bucket '{bucket_name}' does not exist. "
                        f"Please create it or specify a different bucket using "
                        f"VERTEX_BATCH_BUCKET environment variable or gcs_bucket model arg."
                    )
            except Exception as e:
                if "does not exist" in str(e):
                    raise
                # If exists() check fails for other reasons, continue and let upload fail with clearer error
                logger.warning(f"Could not verify bucket existence: {e}")

            blob = bucket.blob(gcs_path)

            # Read file content
            temp_file.seek(0)
            content = temp_file.read()

            logger.info(f"Uploading {len(content)} bytes to {gcs_uri}")

            # Upload to GCS in a thread pool to avoid blocking
            # Use cancellable=True to allow interruption
            await anyio.to_thread.run_sync(
                functools.partial(
                    blob.upload_from_string, content, content_type="application/jsonl"
                ),
                cancellable=True,
            )

            logger.info(f"Successfully uploaded batch file to {gcs_uri}")
            return gcs_uri

        except Exception as e:
            logger.error(f"Failed to upload batch file to GCS: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to upload batch file to GCS bucket '{bucket_name}': {e}"
            ) from e

    async def _find_predictions_blob(self, gcs_uri: str) -> str:
        """Find predictions.jsonl in the GCS output directory.

        Vertex AI batch outputs follow the pattern:
        gs://.../dest/prediction-model-{timestamp}/predictions.jsonl

        The gcs_uri from job.dest.gcs_uri is a prefix, not the exact file path.
        We need to list blobs and find predictions.jsonl
        """
        import logging

        import anyio

        logger = logging.getLogger(__name__)

        try:
            if not gcs_uri.startswith("gs://"):
                raise ValueError(f"Invalid GCS URI: {gcs_uri}")

            gcs_client = self._get_gcs_client()

            # Remove gs:// prefix and split bucket/path
            # Example: gs://bucket/path/to/output -> bucket, path/to/output
            path_without_prefix = gcs_uri[5:]
            parts = path_without_prefix.split("/", 1)
            bucket_name = parts[0]
            base_path = parts[1] if len(parts) > 1 else ""

            bucket = gcs_client.bucket(bucket_name)

            def find_predictions():
                try:
                    # List all blobs with this prefix
                    blobs_list = list(bucket.list_blobs(prefix=base_path))

                    logger.info(
                        f"Searching for predictions.jsonl in gs://{bucket_name}/{base_path}"
                    )
                    logger.info(f"Found {len(blobs_list)} blobs with prefix")

                    if not blobs_list:
                        logger.warning(f"No blobs found with prefix: {base_path}")
                        return None

                    # Sort by creation time (most recent first)
                    blobs = sorted(
                        blobs_list,
                        key=lambda x: x.time_created if x.time_created else 0,
                        reverse=True,
                    )

                    # Look for predictions.jsonl
                    for blob in blobs:
                        logger.debug(f"  Checking blob: {blob.name}")
                        if blob.name.endswith("predictions.jsonl"):
                            predictions_uri = f"gs://{bucket_name}/{blob.name}"
                            logger.info(f"Found predictions file: {predictions_uri}")
                            return predictions_uri

                    # Log all blob names for debugging
                    logger.warning("No predictions.jsonl found. Available files:")
                    for blob in blobs[:10]:  # Show first 10
                        logger.warning(f"  - {blob.name}")
                    if len(blobs) > 10:
                        logger.warning(f"  ... and {len(blobs) - 10} more files")

                    return None

                except Exception as e:
                    logger.error(
                        f"Error listing blobs in gs://{bucket_name}/{base_path}: {e}"
                    )
                    raise

            predictions_uri = await anyio.to_thread.run_sync(
                find_predictions, cancellable=True
            )

            if not predictions_uri:
                raise RuntimeError(
                    f"No predictions.jsonl file found in {gcs_uri}. "
                    f"The batch job may still be running, failed before writing results, "
                    f"or the output was written to a different location. "
                    f"Check the Vertex AI console for job status."
                )

            return predictions_uri

        except Exception as e:
            logger.error(f"Failed to find predictions blob: {e}", exc_info=True)
            if isinstance(e, RuntimeError) and "No predictions.jsonl" in str(e):
                raise
            raise RuntimeError(
                f"Failed to search for predictions file in {gcs_uri}: {e}"
            ) from e

    @override
    async def _download_result_file(self, file_uri: str) -> bytes:
        """Download result file from GCS URI.

        For Vertex AI batch processing, file_uri is typically a prefix/directory path
        from job.dest.gcs_uri. We need to find the actual predictions.jsonl file
        within that path.
        """
        import logging

        import anyio

        logger = logging.getLogger(__name__)

        try:
            # Vertex AI batch results should always be GCS URIs
            if not file_uri.startswith("gs://"):
                raise ValueError(
                    f"Expected GCS URI (gs://...) for Vertex AI batch results, got: {file_uri}. "
                    "Ensure your Vertex AI batch job is configured to output results to GCS."
                )

            # If file_uri doesn't end with .jsonl, it's likely a directory prefix
            # We need to search for predictions.jsonl
            if not file_uri.endswith(".jsonl"):
                logger.info(
                    "file_uri is a directory path, searching for predictions.jsonl"
                )
                file_uri = await self._find_predictions_blob(file_uri)

            gcs_client = self._get_gcs_client()

            # Remove gs:// prefix and split bucket/path
            path_parts = file_uri[5:].split("/", 1)
            if len(path_parts) != 2:
                raise ValueError(f"Invalid GCS URI format: {file_uri}")

            bucket_name, blob_path = path_parts

            # Get bucket and blob
            bucket = gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            logger.info(f"Downloading batch results from: {file_uri}")

            # Check if blob exists
            blob_exists = await anyio.to_thread.run_sync(blob.exists, cancellable=True)
            if not blob_exists:
                raise RuntimeError(
                    f"Result file does not exist in GCS: {file_uri}. "
                    f"The batch job may have failed or the results were not written."
                )

            # Download content in a thread pool
            content = await anyio.to_thread.run_sync(
                blob.download_as_bytes, cancellable=True
            )

            # Log file size for debugging
            logger.info(f"Downloaded {len(content)} bytes from {file_uri}")

            # Validate it's proper JSONL by checking lines
            try:
                import json

                lines = content.decode("utf-8").split("\n")
                valid_lines = [line for line in lines if line.strip()]
                logger.info(f"File contains {len(valid_lines)} non-empty lines")

                # Check for malformed JSON lines
                malformed_count = 0
                for i, line in enumerate(valid_lines[:10]):  # Check first 10 lines
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        malformed_count += 1
                        logger.warning(f"Line {i + 1} has malformed JSON: {e}")
                        logger.warning(f"Problematic line preview: {line[:200]}...")

                if malformed_count > 0:
                    logger.warning(
                        f"Found {malformed_count} malformed lines in first 10 lines"
                    )

            except Exception as e:
                logger.warning(f"Error validating JSONL format: {e}")

            return content

        except Exception as e:
            logger.error(f"Failed to download result file from GCS: {e}", exc_info=True)
            if isinstance(e, RuntimeError) and "does not exist" in str(e):
                raise
            raise RuntimeError(
                f"Failed to download batch results from {file_uri}: {e}"
            ) from e

    def _clean_vertex_response(
        self, response_data: dict[str, pydantic.JsonValue]
    ) -> dict[str, pydantic.JsonValue]:
        """Remove Vertex AI specific fields that aren't in the standard GenerateContentResponse schema.

        Vertex AI batch responses include additional fields like:
        - candidates.*.score (float)
        - usageMetadata.billablePromptUsage (dict)

        These need to be filtered out before pydantic validation.
        """
        # Deep copy to avoid modifying original
        import copy

        cleaned = copy.deepcopy(response_data)

        # Remove score from candidates
        if "candidates" in cleaned and isinstance(cleaned["candidates"], list):
            for candidate in cleaned["candidates"]:
                if isinstance(candidate, dict) and "score" in candidate:
                    del candidate["score"]

        # Remove billablePromptUsage from usageMetadata
        if "usageMetadata" in cleaned and isinstance(cleaned["usageMetadata"], dict):
            if "billablePromptUsage" in cleaned["usageMetadata"]:
                del cleaned["usageMetadata"]["billablePromptUsage"]

        return cleaned

    @override
    async def _parse_result_file(
        self, file_uri: str
    ) -> dict[str, GenerateContentResponse | Exception]:
        """Override to handle malformed JSON in Vertex AI batch responses.

        Base class uses dict comprehension with json.loads() which fails entirely
        on first malformed line. This version uses explicit loop with try/except
        and provides detailed error information.
        """
        import json
        import logging
        import re

        logger = logging.getLogger(__name__)

        content = await self._download_result_file(file_uri)
        lines = content.decode("utf-8").splitlines()

        results: dict[str, GenerateContentResponse | Exception] = {}
        skipped_lines = 0
        skipped_custom_ids: list[str] = []

        for line_num, line in enumerate(lines, 1):
            if not line.strip():
                continue

            try:
                line_data = json.loads(line)
                custom_id, response_or_exception = self._parse_jsonl_line(line_data)
                if custom_id:
                    results[custom_id] = response_or_exception
            except json.JSONDecodeError as e:
                skipped_lines += 1
                logger.warning(f"Malformed JSON on line {line_num}: {str(e)[:100]}")
                logger.warning(f"Problematic line (first 500 chars): {line[:500]}...")

                # Try to extract custom_id and record error
                try:
                    if '"custom_id"' in line:
                        match = re.search(r'"custom_id"\s*:\s*"([^"]+)"', line)
                        if match:
                            custom_id = match.group(1)
                            skipped_custom_ids.append(custom_id)
                            # Record as RuntimeError with detailed info
                            results[custom_id] = RuntimeError(
                                f"Malformed JSON in batch response (line {line_num}): {str(e)[:200]}"
                            )
                            logger.warning(
                                f"  -> Recorded error for custom_id: {custom_id}"
                            )
                        else:
                            logger.warning(
                                "  -> Could not extract custom_id from malformed line"
                            )
                    else:
                        # Likely a continuation line from previous malformed JSON
                        logger.debug(
                            "  -> Line appears to be continuation of previous malformed JSON"
                        )
                except Exception:
                    pass

                continue

            except Exception as e:
                skipped_lines += 1
                logger.warning(f"Error processing line {line_num}: {str(e)[:100]}")
                logger.warning(f"Problematic line (first 500 chars): {line[:500]}...")

                # Try to extract custom_id and record error
                try:
                    if '"custom_id"' in line:
                        match = re.search(r'"custom_id"\s*:\s*"([^"]+)"', line)
                        if match:
                            custom_id = match.group(1)
                            skipped_custom_ids.append(custom_id)
                            # Record as RuntimeError with detailed info
                            results[custom_id] = RuntimeError(
                                f"Error processing batch response (line {line_num}): {str(e)[:200]}"
                            )
                            logger.warning(
                                f"  -> Recorded error for custom_id: {custom_id}"
                            )
                        else:
                            logger.warning(
                                "  -> Could not extract custom_id from malformed line"
                            )
                    else:
                        # Likely a continuation line from previous malformed JSON
                        logger.debug(
                            "  -> Line appears to be continuation of previous malformed JSON"
                        )
                except Exception:
                    pass

                continue

        error_count = sum(1 for v in results.values() if isinstance(v, Exception))
        success_count = len(results) - error_count

        if skipped_lines > 0:
            logger.warning(
                f"Skipped {skipped_lines} malformed lines out of {len(lines)} total. "
                f"Parsed {len(results)} results: {success_count} successful, {error_count} errors."
            )
            if skipped_custom_ids:
                logger.warning(
                    f"Malformed responses for custom_ids: {', '.join(skipped_custom_ids[:5])}"
                    + (
                        f" ... and {len(skipped_custom_ids) - 5} more"
                        if len(skipped_custom_ids) > 5
                        else ""
                    )
                )
        else:
            logger.info(
                f"Successfully parsed {len(results)} results from {len(lines)} lines"
            )

        return results

    @override
    def _parse_jsonl_line(
        self, line_data: dict[str, pydantic.JsonValue]
    ) -> tuple[str, GenerateContentResponse | Exception]:
        """Parse a single JSONL result line from Vertex AI.

        Follows the OpenAI pattern: extract custom_id first, then process
        response or error. Exceptions are allowed to propagate naturally
        to be handled by _parse_result_file.
        """
        # Extract custom_id - raise ValueError if missing (like OpenAI does)
        # Vertex AI batch uses "custom_id", but also check "key" for compatibility
        custom_id = line_data.get("custom_id") or line_data.get("key")
        if not custom_id or not isinstance(custom_id, str):
            raise ValueError(
                f"Unable to find custom_id in batched request result. Keys: {list(line_data.keys())}"
            )

        # Check for error response
        if "error" in line_data:
            error_data = JobError.model_validate(line_data["error"])
            return (
                custom_id,
                RuntimeError(f"{error_data.message} (code: {error_data.code})"),
            )

        # Success response - clean Vertex-specific fields and validate
        response_data = self._clean_vertex_response(line_data["response"])
        return custom_id, GenerateContentResponse.model_validate(response_data)

    @override
    def _uris_from_completion_info(
        self, completion_info: CompletedBatchInfo
    ) -> list[str]:
        return [completion_info]

    @override
    async def _submit_batch_for_file(
        self, file_id: str, extra_headers: dict[str, str]
    ) -> str:
        """Submit a batch job using the GCS file URI.

        Args:
            file_id: GCS URI of the uploaded batch file (gs://bucket/path)
            extra_headers: Headers to include in batch creation request

        Returns:
            Batch job ID/name from Vertex AI
        """
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Extract request ID for batch job display name if available
            request_id = extra_headers.get(HttpxHooks.REQUEST_ID_HEADER, "")
            display_name = (
                f"inspect_ai_batch_job_{request_id}"
                if request_id
                else f"inspect_ai_batch_job_{int(time.time())}"
            )

            config = CreateBatchJobConfig(
                display_name=display_name,
                http_options=HttpOptions(headers=extra_headers or None),
            )

            logger.info(
                f"Submitting batch job with model={self._model_name}, src={file_id}"
            )

            # Use the google.genai Client batch API with the GCS URI
            # The Vertex AI backend should accept GCS URIs directly
            batch_job = await self._client.aio.batches.create(
                model=self._model_name,
                src=file_id,  # This is the GCS URI
                config=config,
            )

            job_name = batch_job.name or ""
            logger.info(f"Successfully created batch job: {job_name}")
            return job_name

        except Exception as e:
            logger.error(f"Failed to submit batch job: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to submit Vertex AI batch job for file {file_id}: {e}"
            ) from e

    # Batcher overrides

    @override
    async def _check_batch(
        self, batch: Batch[GenerateContentResponse]
    ) -> BatchCheckResult[CompletedBatchInfo]:
        """Check the status of a batch job."""
        batch_job = await self._client.aio.batches.get(name=batch.id)

        created_at = int(
            (
                batch_job.create_time
                if batch_job.create_time
                else datetime.now(tz=timezone.utc)
            ).timestamp()
        )

        # Handle different job states
        if (
            batch_job.state == JobState.JOB_STATE_PENDING
            or batch_job.state == JobState.JOB_STATE_RUNNING
        ):
            return BatchCheckResult(
                completed_count=0,
                failed_count=0,
                created_at=created_at,
                completion_info=None,
            )
        elif batch_job.state == JobState.JOB_STATE_SUCCEEDED:
            # The dest should contain a GCS URI for Vertex AI batch results
            assert batch_job.dest, "must find batch dest"

            # For Vertex AI with GCS, we expect gcs_uri in the destination
            result_uri = None
            if hasattr(batch_job.dest, "gcs_uri") and batch_job.dest.gcs_uri:
                result_uri = batch_job.dest.gcs_uri
            elif hasattr(batch_job.dest, "file_name") and batch_job.dest.file_name:
                # If file_name is provided instead of gcs_uri, it might be a GCS path
                # Try to construct the GCS URI if it looks like a path
                file_name = batch_job.dest.file_name
                if file_name.startswith("gs://"):
                    result_uri = file_name
                else:
                    # If it's not a GCS URI, we can't handle it with this implementation
                    raise RuntimeError(
                        f"Batch job succeeded but destination is not a GCS URI: {file_name}. "
                        "Vertex AI batch processing with GCS requires results in GCS."
                    )
            else:
                raise RuntimeError(
                    f"Batch job succeeded but no valid GCS destination found: {batch_job.dest}"
                )

            return BatchCheckResult(
                completed_count=len(
                    batch.requests
                ),  # Assume all completed if succeeded
                failed_count=0,  # Failed count will be determined during result parsing
                created_at=created_at,
                completion_info=result_uri,
            )
        elif batch_job.state in [
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
        ]:
            # Job failed or was cancelled - but check if there's a results file with partial results
            # Even failed jobs can have some successful responses that should be processed
            result_uri = None
            if batch_job.dest:
                if hasattr(batch_job.dest, "gcs_uri") and batch_job.dest.gcs_uri:
                    result_uri = batch_job.dest.gcs_uri
                elif hasattr(batch_job.dest, "file_name") and batch_job.dest.file_name:
                    file_name = batch_job.dest.file_name
                    if file_name.startswith("gs://"):
                        result_uri = file_name

            if result_uri:
                # We have a results file, process it to get successful responses
                # Failed count will be determined during result parsing
                return BatchCheckResult(
                    completed_count=0,  # Will be updated when parsing results
                    failed_count=0,  # Will be updated when parsing results
                    created_at=created_at,
                    completion_info=result_uri,
                )
            else:
                # No results file available - all requests failed
                return BatchCheckResult(
                    completed_count=0,
                    failed_count=len(batch.requests),
                    created_at=created_at,
                    completion_info=None,
                )
        else:
            # Unknown state - treat as pending
            return BatchCheckResult(
                completed_count=0,
                failed_count=0,
                created_at=created_at,
                completion_info=None,
            )
