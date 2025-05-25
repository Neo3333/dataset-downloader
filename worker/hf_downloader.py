import os
import logging

from huggingface_hub import snapshot_download # type: ignore
from config import (
  HF_HUB_TOKEN,
  FILERESTORE_MOUNT_PATH,
  GCS_BUCKET,
  GCS_PREFIX,
)
from google.cloud import storage

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# GCS Client
_storage_client = storage.Client()

def download_dataset(
  repo_id: str,
  config: str | None = None,
  split: str | None = None,
  dest: str = FILERESTORE_MOUNT_PATH
) -> None:
  """
  Download a dataset from HuggingFace Hub to the specified destination directory.
  Optionally filter by config name or split using allow_patterns.
  """
  # 1) Download into Filestore
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

  logger.info(f"Downloading dataset {repo_id} to {dest}...")
  snapshot_download(**snapshot_kwargs)
  logger.info("Download complete")

  # 2) Upload from Filestore to GCS
  bucket = _storage_client.bucket(GCS_BUCKET)
  for root, _, files in os.walk(dest):
    for fname in files:
      local_path = os.path.join(root, fname)
      rel_path   = os.path.relpath(local_path, dest)
      gcs_path   = os.path.join(GCS_PREFIX, repo_id, rel_path).lstrip("/")

      blob = bucket.blob(gcs_path)
      blob.upload_from_filename(local_path)
      logger.info(f"Uploaded gs://{GCS_BUCKET}/{gcs_path}")

  logger.info("All files uploaded to GCS.")