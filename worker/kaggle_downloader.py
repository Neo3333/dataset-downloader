import os
import sys
import logging
import json
import subprocess
import time
from pathlib import Path

from config import (
  KAGGLE_KEY,
  KAGGLE_USERNAME,
  FILERESTORE_MOUNT_PATH,
  GCS_KAGGLE_PREFIX,
  GCS_BUCKET,
  UPLOAD_WORKERS,
  CHUNK_SIZE_MB,
  GOOGLE_CLOUD_PROJECT,
  PUBSUB_TOPIC,
)
from util.kaggle import get_all_dataset_files
from util.status import Status
from gcs.gcs_uploader import upload_files
from kaggle.api.kaggle_api_extended import KaggleApi # type: ignore
from tqdm import tqdm # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed
from pubsub.publish import Publisher

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

publisher = Publisher(project=GOOGLE_CLOUD_PROJECT, topic=PUBSUB_TOPIC)

_kaggle_api = None
if KAGGLE_USERNAME and KAGGLE_KEY:
  api = KaggleApi()
  api.authenticate()
  _kaggle_api = api


MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2

def _ensure_kaggle_credentials():
  """
  If KAGGLE_USERNAME and KAGGLE_KEY are set in env vars,
  write them to /root/.kaggle/kaggle.json so that 'kaggle' CLI can authenticate.
  """
  if not KAGGLE_USERNAME or not KAGGLE_KEY:
    raise ValueError('Insufficient Kaggle credentials')

  cred_dir = Path("/root/.kaggle")
  cred_dir.mkdir(parents=True, exist_ok=True)
  cred_file = cred_dir / "kaggle.json"

  # Only write if it doesn't exist or if contents differ
  new_contents = {"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}
  if cred_file.exists():
    try:
      current = json.loads(cred_file.read_text())
      if current == new_contents:
        return
    except Exception:
        pass

  cred_file.write_text(json.dumps(new_contents))
  cred_file.chmod(0o600)

def _download_file_worker(repo_id: str, filename: str, dest: str, kaggle_api_instance) -> Status:
  """
  Worker function to download a single file from a Kaggle dataset.
  This function is designed to be executed by a thread pool executor.
  
  Args:
    repo_id: The Kaggle repository ID (e.g., 'owner/dataset').
    filename: The name of the file to download.
    dest: The destination directory to save the file.
    kaggle_api_instance: An instance of the Kaggle API.

  Returns:
    Status of the download.
  """
  for attempt in range(MAX_RETRIES):
    try:
      # The Kaggle API client is generally thread-safe for I/O operations.
      kaggle_api_instance.dataset_download_file(
        repo_id,
        filename,
        path=dest,
        force=False,
        quiet=True,  # Suppress verbose output for each file to keep the console clean
      )
      return Status(ok=True)
    except Exception as e:
      # Check if the exception message contains '429', indicating a rate limit error.
      if '429' in str(e):
        if attempt < MAX_RETRIES - 1:
          # Calculate sleep time with exponential backoff (e.g., 2, 4, 8 seconds)
          sleep_time = BASE_BACKOFF_SECONDS * (2 ** attempt)
          logging.warning(
            f"Rate limit hit for '{filename}'. Retrying in {sleep_time}s... (Attempt {attempt + 2}/{MAX_RETRIES})"
          )
          time.sleep(sleep_time)
        else:
          # Log an error if all retries fail
          message=f"Failed to download '{filename}' after {MAX_RETRIES} attempts due to rate limiting."
          logging.error(message)
          return Status(ok=False, message=message)
      else:
        # For any other exception, log the error and fail immediately.
        message = f"Failed to download '{filename}'. Unrelated Error: {e}"
        logging.error(message)
        return Status(ok=False, message=message)

def download_kaggle_dataset_concurrently(repo_id: str, dest_suffix: str, max_workers: int = 10) -> None:
  """
  Uses the Kaggle API to concurrently download all files from a dataset
  into the specified destination, then uploads them.

  Args:
    repo_id: The Kaggle repository ID (e.g., 'owner/dataset').
    dest_suffix: A suffix to append to the base destination path.
    max_workers: The maximum number of concurrent download threads.
  """
  global _kaggle_api # Assuming _kaggle_api is a global or accessible instance

  if _kaggle_api is None:
    raise RuntimeError("Kaggle credentials not set or Kaggle API not initialized.")

  base_dest = FILERESTORE_MOUNT_PATH
  dest = os.path.join(base_dest, dest_suffix) if dest_suffix else base_dest

  logging.info(f"Preparing to download Kaggle dataset '{repo_id}' to {dest}...")
  os.makedirs(dest, exist_ok=True)

  logging.info(f"Listing files in Kaggle dataset '{repo_id}'...")
  repo_id_comp = repo_id.split('/')
  if not repo_id_comp or len(repo_id_comp) != 2:
    raise ValueError(f"Invalid repo_id format: '{repo_id}'. Expected 'owner/dataset'.")

  try:
    # Assumes this function returns a list of file metadata dictionaries
    all_files = get_all_dataset_files(
      repo_id_comp[0], repo_id_comp[1], KAGGLE_USERNAME, KAGGLE_KEY
    )
    if not all_files:
      logging.warning(f"No files found for dataset '{repo_id}'.")
      return
  except Exception as e:
    logging.error(f"Error encountered while getting the file list: {e}")
    raise

  logging.info(f"Retrieved {len(all_files)} files in total")

  logging.info("Start downloading")
  # Use ThreadPoolExecutor to download files in parallel
  all_file_names = [f.get('name', 'N/A') for f in all_files]
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit a download task for each file to the executor
    futures = [
      executor.submit(_download_file_worker, repo_id, file_name, dest, _kaggle_api)
      for file_name in all_file_names
    ]

    # Use tqdm to create a progress bar that updates as each download completes
    progress_bar = tqdm(as_completed(futures), total=len(futures), desc=f"Downloading '{repo_id}'")
    
    for future in progress_bar:
      try:
        # The result() call will re-raise any exceptions from the worker thread
        status = future.result()
        if not status.is_ok():
          # The error is already logged in the worker, but you could add more handling here.
          pass
      except Exception as exc:
        logging.error(f"An exception occurred for: {exc}")

  logging.info("Kaggle concurrent download process has finished.")

  try:
    logging.info(f"Starting upload from {dest} to GCS...")
    upload_files(
      source=dest,
      bucket=GCS_BUCKET,
      repo_id=repo_id,
      dest_prefix=GCS_KAGGLE_PREFIX,
      upload_worker=UPLOAD_WORKERS,
      chunk_size_mb=CHUNK_SIZE_MB
    )
    logging.info("Upload to GCS complete.")
  except Exception as e:
    # Changed logger to logging to maintain consistency
    logging.error(f"An exception was encountered during the GCS upload: {e}")
    raise

def download_kaggle_dataset(repo_id: str, dest_suffix: str) -> None:
  """
  Use Kaggle API to download entire dataset (owner/dataset) into `dest`, then upload the files.
  """
  if _kaggle_api is None:
    raise RuntimeError("Kaggle credentials not set or Kaggle API not initialized.")

  base_dest = FILERESTORE_MOUNT_PATH
  dest = os.path.join(base_dest, dest_suffix) if dest_suffix else base_dest

  logging.info(f"Downloading Kaggle dataset '{repo_id}' to {dest}...")
  os.makedirs(dest, exist_ok=True)

  logging.info(f"Listing files in Kaggle dataset '{repo_id}'...")
  repo_id_comp = repo_id.split('/')
  if not repo_id_comp or len(repo_id_comp) != 2:
    raise ValueError(f'Invalid repo_id {repo_id}')

  try:
    all_files = get_all_dataset_files(
      repo_id_comp[0], repo_id_comp[1], KAGGLE_USERNAME, KAGGLE_KEY)
  except Exception as e:
    logging.error(f"Error {e} encountered while getting full file list.")
    raise e

  for f in tqdm(all_files):
    filename = f.get('name', 'N/A')
    try:
      _kaggle_api.dataset_download_file(
        repo_id,
        filename,
        path=dest,
        force=True,
        quiet=False,
      )
    except Exception as e:
      logging.error(f"Exception encountered {e}")
      continue

  logging.info("Kaggle download complete.")

  try:
    upload_files(
      source=dest,
      bucket=GCS_BUCKET,
      repo_id=repo_id,
      dest_prefix=GCS_KAGGLE_PREFIX,
      upload_worker=UPLOAD_WORKERS,
      chunk_size_mb=CHUNK_SIZE_MB
    )
  except Exception as e:
    logger.error(f"Exception encountered while uploading to GCS: {e}")
    raise

def download_kaggle_dataset_with_cli(repo_id: str, dest_suffix: str) -> None:
  """
  Use Kaggle CLI to download entire dataset (owner/dataset) into `dest`, then upload the files.
  """
  try:
    _ensure_kaggle_credentials()
  except Exception as e:
    logging.error(f'Exception encountered while setting up local Kaggle credentials {e}')
    raise

  base_dest = FILERESTORE_MOUNT_PATH
  dest = os.path.join(base_dest, dest_suffix) if dest_suffix else base_dest

  logging.info(f"Downloading Kaggle dataset '{repo_id}' to {dest} via CLI…")
  os.makedirs(dest, exist_ok=True)

  # Build the 'kaggle datasets download' command.
  # -p DEST    → download into DEST
  # --unzip    → unzip in DEST
  cmd = [
    "kaggle", "datasets", "download",
    repo_id,
    "-p", dest,
  ]
  # Run Kaggle CLI; it will show a progress bar on stdout/stderr
  try:
    with subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr) as proc:
      proc.communicate()
    logging.info("Kaggle download completed.")
  except subprocess.CalledProcessError as e:
    raise RuntimeError(f"Kaggle CLI failed: {e}") from e

  logging.info("Kaggle download complete.")
  try:
    gcs_dest = upload_files(
      source=dest,
      bucket=GCS_BUCKET,
      repo_id=repo_id,
      dest_prefix=GCS_KAGGLE_PREFIX,
      upload_worker=UPLOAD_WORKERS,
      chunk_size_mb=CHUNK_SIZE_MB
    )
  except Exception as e:
    logger.error(f"Exception encountered while uploading to GCS: {e}")
    raise

  status = publisher.publish(dataset=repo_id, destination=gcs_dest)
  logging.info(f"Publish status {status}")
  
  