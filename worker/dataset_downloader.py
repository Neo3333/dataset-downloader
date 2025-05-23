import os

from huggingface_hub import snapshot_download # type: ignore
from config import HF_HUB_TOKEN, FILERESTORE_MOUNT_PATH


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

  print(f"Downloading dataset {repo_id} to {dest}...")
  snapshot_download(**snapshot_kwargs)
  print("Download complete.")