import os
import shutil
import logging

from config import FILERESTORE_MOUNT_PATH

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def delete_all_files():
  mount_path = FILERESTORE_MOUNT_PATH
  for entry in os.listdir(mount_path):
    full_path = os.path.join(mount_path, entry)
    try:
      if os.path.isdir(full_path):
        shutil.rmtree(full_path)
      else:
        os.remove(full_path)
      logging.info(f"Deleted: {full_path}")
    except Exception as e:
      logging.info(f"Failed to delete {full_path}: {e}")

if __name__ == "__main__":
  delete_all_files()