import logging
from typing import Tuple

from google.cloud.run_v2.services.jobs import JobsClient # type: ignore
from google.cloud.run_v2.types import RunJobRequest # type: ignore
from google.api_core.exceptions import GoogleAPICallError, RetryError # type: ignore

from config import JOB_RESOURCE, SERVICE_ACCOUNT
from util.status import Status

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize Jobs client
_jobs_client = JobsClient()


def trigger_download_job(dataset: str, dest_suffix: str = "") -> Tuple[str, Status]:
  """
  Launches a Cloud Run Job execution asynchronously.
  Returns the execution name.
  """

  # Build container overrides: args include optional suffix
  args = ["--dataset", dataset]
  if dest_suffix:
    args += ["--dest_suffix", dest_suffix]
  container_override = {
    "args": args
  }

  request = RunJobRequest(
    name=JOB_RESOURCE,
    overrides={"container_overrides": [container_override]}
  )
  logger.info(f"Triggering Cloud Run Job: {JOB_RESOURCE}")
  try:
    op = _jobs_client.run_job(request=request)
    operation_id = op.operation.name
    logger.info(f"Cloud Run operation started: {operation_id}")
    return operation_id, Status(ok=True)
  except GoogleAPICallError as e:
    logger.error(f"Google API error: {e.message} (code: {e.code})")
    return None, Status(ok=False, message=e.message, code=e.code)
  except RetryError as e:
    logger.error(f"Retry error: {e}")
    return None, Status(ok=False, message=str(e))
  except ValueError as e:
    return None, Status(ok=False, message=e.message)