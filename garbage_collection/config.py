import os
from dotenv import load_dotenv # type: ignore

# Load environment variables from .env
load_dotenv()

# Default mount path for Filestore
FILERESTORE_MOUNT_PATH = os.getenv("FILERESTORE_MOUNT_PATH", "/mnt/filestore")