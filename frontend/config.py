import os
from dotenv import load_dotenv # type: ignore

# Load environment variables from .env
load_dotenv()

# GCP and Cloud Run settings
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("CLOUD_RUN_REGION", "us-central1")
JOB_NAME = os.getenv("DOWNLOAD_JOB_NAME", "dataset-downloader")

# Full resource name for the job
JOB_RESOURCE = f"projects/{PROJECT_ID}/locations/{LOCATION}/jobs/{JOB_NAME}"

# Service account email for OIDC token
SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT_EMAIL")

# Flask settings
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("PORT", 8080))