import os
from dotenv import load_dotenv # type: ignore

# Load environment variables from .env
load_dotenv()

# Hugging Face Hub token environment variable
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN")

# Default mount path for Filestore
FILERESTORE_MOUNT_PATH = os.getenv("FILERESTORE_MOUNT_PATH", "/mnt/filestore")