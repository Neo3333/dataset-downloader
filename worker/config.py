import os
from dotenv import load_dotenv # type: ignore

# Load environment variables from .env
load_dotenv()

# Hugging Face Hub token environment variable
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN")

# Kaggle
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
KAGGLE_DOWNLOAD_WORKER = os.getenv("KAGGLE_DOWNLOAD_WORKER", 3)

# Default mount path for Filestore
FILERESTORE_MOUNT_PATH = os.getenv("FILERESTORE_MOUNT_PATH", "/mnt/filestore")

# GCS
GCS_BUCKET = os.getenv("GCS_BUCKET", "3p-datasets-bucket")
GCS_HUGGING_FACE_PREFIX = os.getenv("GCS_HUGGING_FACE_PREFIX", "huggingface")  # optional sub-folder
GCS_KAGGLE_PREFIX = os.getenv("GCS_KAGGLE_PREFIX", "kaggle")  # optional sub-folder

# Upload tuning
UPLOAD_WORKERS = int(os.getenv("UPLOAD_WORKERS", "10"))   # default 4 threads
CHUNK_SIZE_MB  = int(os.getenv("CHUNK_SIZE_MB",  "64"))   # default 8 MiB