import time
import logging
import os
from config import (
  GCS_BUCKET,
  UPLOAD_WORKERS,
  CHUNK_SIZE_MB,
)

from google.cloud import storage
from tqdm import tqdm # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# GCS Client
_storage_client = storage.Client()

def _upload_one(
    bucket: storage.Bucket,
    local_path: str, 
    gcs_path: str, 
    max_retries: int=3
) -> None:
  """
    Upload a file to a GCS bucket with retries and exponential backoff.

    Args:
      bucket (storage.Bucket): The target GCS bucket.
      local_path (str): Path to the local file.
      gcs_path (str): Target path in the GCS bucket.
      max_retries (int): Number of retry attempts on failure.
  """
  blob = bucket.blob(gcs_path)
  blob.chunk_size = CHUNK_SIZE_MB * 1024 * 1024
  for attempt in range(max_retries):
    try:
      blob.upload_from_filename(local_path)
      return
    except Exception as e:
      if attempt < max_retries - 1:
        time.sleep(2 ** attempt)  # Exponential backoff
      else:
        raise

def upload_files(
    source: str,
    repo_id: str,
    dest_prefix: str,
    parquet_only: bool = False) -> None:
  """
  Walk `source`, find all files, and upload to GCS in parallel.
  """
  bucket = _storage_client.bucket(GCS_BUCKET)
  # Tuples (filestore, gcs) of file locations to be uploaded
  to_upload = []
  for root, dirs, files in os.walk(source):
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
      rel_path = os.path.relpath(local_path, source)
      gcs_path = os.path.join(dest_prefix, repo_id, rel_path).lstrip("/")
      to_upload.append((local_path, gcs_path))

  # Log the first up to 10 files to upload
  sample = to_upload[:10]
  logger.info("First files to upload:")
  for lp, gp in sample:
    logger.info(f" - {lp} -> gs://{GCS_BUCKET}/{gp}")

  logger.info(f"Uploading {len(to_upload)} parquet files to GCS in parallel…")
  logger.info(f"Uploading files to GCS with {UPLOAD_WORKERS} workers and {CHUNK_SIZE_MB}MB chunks...")

  # Parallel upload
  with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as executor:
    futures = [
      executor.submit(_upload_one, bucket, lp, gp)
      for lp, gp in to_upload
    ]
    for f in tqdm(as_completed(futures), total=len(futures)):
      try:
        f.result()
      except Exception as e:
        logging.error("Upload error:", e)

  logger.info("All files uploaded to GCS.")