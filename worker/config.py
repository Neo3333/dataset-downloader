import os

# Hugging Face Hub token environment variable
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN")

# Default mount path for Filestore
FILERESTORE_MOUNT_PATH = os.getenv("FILERESTORE_MOUNT_PATH", "/mnt/filestore")

# Flask settings
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("PORT", 8080))