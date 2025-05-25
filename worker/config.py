import os
from dotenv import load_dotenv # type: ignore

# Load environment variables from .env
load_dotenv()

# Hugging Face Hub token environment variable
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN")

# Default mount path for Filestore
FILERESTORE_MOUNT_PATH = os.getenv("FILERESTORE_MOUNT_PATH", "/mnt/filestore")

# GCS
GCS_BUCKET = os.getenv("GCS_BUCKET", "3p-datasets-bucket")
GCS_PREFIX = os.getenv("GCS_PREFIX", "huggingface")  # optional sub-folder

# Upload tuning
UPLOAD_WORKERS = int(os.getenv("UPLOAD_WORKERS", "4"))   # default 4 threads
CHUNK_SIZE_MB  = int(os.getenv("CHUNK_SIZE_MB",  "8"))   # default 8 MiB