import os
import logging
import time

from huggingface_hub import snapshot_download # type: ignore
from config import (
  HF_HUB_TOKEN,
  FILERESTORE_MOUNT_PATH,
  GCS_BUCKET,
  GCS_HUGGING_FACE_PREFIX,
  UPLOAD_WORKERS,
  CHUNK_SIZE_MB,
  GOOGLE_CLOUD_PROJECT,
  PUBSUB_TOPIC,
)
from gcs.gcs_uploader import upload_files
from util.huggingface import check_datasets_server_parquet_status
from pubsub.publish import Publisher

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

publisher = Publisher(project=GOOGLE_CLOUD_PROJECT, topic=PUBSUB_TOPIC)

def download_huggingface_dataset(
  repo_id: str,
  config: str | None = None,
  split: str | None = None,
  dest_suffix: str = "",
  parquet_only: bool = False
) -> None:
  """
  Download a dataset from HuggingFace Hub to the specified destination directory.
  Optionally filter by config name or split using allow_patterns.
  """
  # 1) Download into Filestore

  # Construct the base destination
  base_dest = FILERESTORE_MOUNT_PATH
  # Append suffix if provided
  dest = os.path.join(base_dest, dest_suffix) if dest_suffix else base_dest
  # Ensure destination directory exists
  os.makedirs(dest, exist_ok=True)

  # Build patterns list
  allow_patterns = []
  if config:
    allow_patterns.append(f"*{config}*")
  if split:
    allow_patterns.append(f"*{split}*")
  if not allow_patterns:
    allow_patterns = None  # snapshot_download expects None or sequence

  # Prepare kwargs
  snapshot_kwargs = {
    "repo_id": repo_id,
    "repo_type": "dataset",
    "local_dir": dest,
    # Only apply filter if provided
    **({"allow_patterns": allow_patterns} if allow_patterns is not None else {}),
    **({"token": HF_HUB_TOKEN} if HF_HUB_TOKEN else {})
  }
  parquet_status = check_datasets_server_parquet_status(
    repo_id=repo_id, token=HF_HUB_TOKEN
  )
  # Use the parquet branch since the data split is better presented.
  if parquet_status['available'] and not parquet_status['is_partial']:
    snapshot_kwargs['revision'] = 'refs/convert/parquet'

  logger.info(f"Downloading dataset {repo_id} to {dest}...")

  try:
    snapshot_download(**snapshot_kwargs)
  except Exception as e:
    logger.error(f"Failed to download dataset from Hugging Face: {e}")
    raise

  logger.info("Download complete")

  try:
    gcs_dest = upload_files(
      source=dest,
      bucket=GCS_BUCKET,
      repo_id=repo_id,
      dest_prefix=GCS_HUGGING_FACE_PREFIX,
      upload_worker=UPLOAD_WORKERS,
      chunk_size_mb=CHUNK_SIZE_MB,
      parquet_only=parquet_only
    )
  except Exception as e:
    logger.error(f"Exception encountered while uploading to GCS: {e}")
    raise

  status = publisher.publish(dataset=repo_id, destination=gcs_dest)
  if not status.is_ok():
    logging.error(f"Error encountered while publishing message")