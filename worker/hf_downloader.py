import os
import logging

from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import snapshot_download # type: ignore
from config import (
  HF_HUB_TOKEN,
  FILERESTORE_MOUNT_PATH,
  GCS_BUCKET,
  GCS_PREFIX,
  UPLOAD_WORKERS,
  CHUNK_SIZE_MB,
)
from google.cloud import storage

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# GCS Client
_storage_client = storage.Client()

def _upload_one(bucket, local_path, gcs_path):
  blob = bucket.blob(gcs_path)
  blob.chunk_size = CHUNK_SIZE_MB * 1024 * 1024
  blob.upload_from_filename(local_path)

def download_dataset(
  repo_id: str,
  config: str | None = None,
  split: str | None = None,
  dest: str = FILERESTORE_MOUNT_PATH,
  parquet_only: bool = False
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

  # 2) Excluding temporary directories
  bucket = _storage_client.bucket(GCS_BUCKET)
  # Tuples (filestore, gcs) of file locations to be uploaded
  to_upload = []
  for root, dirs, files in os.walk(dest):
    # Exclude any temp or cache directories
    dirs[:] = [
      d for d in dirs if not (
        d.startswith('.') or
        d.lower().startswith('tmp') or
        d.lower().startswith('temp') or
        d == '__pycache__'
      )
    ]
    for fname in files:
      if parquet_only and not fname.endswith('.parquet'):
        continue
      local_path = os.path.join(root, fname)
      rel_path   = os.path.relpath(local_path, dest)
      gcs_path   = os.path.join(GCS_PREFIX, repo_id, rel_path).lstrip("/")
      to_upload.append((local_path, gcs_path))

  logger.info(f"Uploading {len(to_upload)} parquet files to GCS in parallel…")

  # 3) Parallel upload
  with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as executor:
    futures = [
      executor.submit(_upload_one, bucket, lp, gp)
      for lp, gp in to_upload
    ]
    for f in as_completed(futures):
      try:
        f.result()
      except Exception as e:
        logging.error("Upload error:", e)

  logger.info("All files uploaded to GCS.")