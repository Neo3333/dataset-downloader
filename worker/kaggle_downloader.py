import os
import logging

from config import (
  KAGGLE_KEY,
  KAGGLE_USERNAME,
  FILERESTORE_MOUNT_PATH,
  GCS_KAGGLE_PREFIX
)
from gcs_uploader import upload_files
from kaggle.api.kaggle_api_extended import KaggleApi # type: ignore

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_kaggle_api = None
if KAGGLE_USERNAME and KAGGLE_KEY:
  api = KaggleApi()
  api.authenticate()
  _kaggle_api = api

def download_kaggle_dataset(repo_id: str, dest_suffix: str) -> None:
  """
  Use Kaggle API to download entire dataset (owner/dataset) into `dest`, then upload .parquet files.
  """
  if _kaggle_api is None:
    raise RuntimeError("Kaggle credentials not set or Kaggle API not initialized.")

  base_dest = FILERESTORE_MOUNT_PATH
  dest = os.path.join(base_dest, dest_suffix) if dest_suffix else base_dest

  logging.info(f"Downloading Kaggle dataset '{repo_id}' to {dest}...")
  os.makedirs(dest, exist_ok=True)
  _kaggle_api.dataset_download_files(
    repo_id,
    path=dest,
    unzip=True,
    quiet=False
  )
  logging.info("Kaggle download complete.")

  try:
    upload_files(source=dest, repo_id=repo_id, dest_prefix=GCS_KAGGLE_PREFIX)
  except Exception as e:
    logger.error(f"Exception encountered while uploading to GCS: {e}")
    raise

  

  

  