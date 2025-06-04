import os
import sys
import logging
import json
import subprocess
from pathlib import Path

from config import (
  KAGGLE_KEY,
  KAGGLE_USERNAME,
  FILERESTORE_MOUNT_PATH,
  GCS_KAGGLE_PREFIX
)
from util.kaggle import get_all_dataset_files
from gcs_uploader import upload_files
from kaggle.api.kaggle_api_extended import KaggleApi # type: ignore
from tqdm import tqdm # type: ignore

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_kaggle_api = None
if KAGGLE_USERNAME and KAGGLE_KEY:
  api = KaggleApi()
  api.authenticate()
  _kaggle_api = api

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

  for f in tqdm(all_files, desc=f"Downloading all files from '{repo_id}'"):
    filename = f.get('name', 'N/A')
    logging.info(f"Downloading file: {filename}")
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
    upload_files(source=dest, repo_id=repo_id, dest_prefix=GCS_KAGGLE_PREFIX)
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
    upload_files(source=dest, repo_id=repo_id, dest_prefix=GCS_KAGGLE_PREFIX)
  except Exception as e:
    logger.error(f"Exception encountered while uploading to GCS: {e}")
    raise
  