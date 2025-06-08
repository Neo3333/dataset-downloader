import logging
import json
import datetime

from google.cloud import pubsub_v1
from concurrent.futures import TimeoutError
from google.api_core import exceptions # type: ignore

from util.status import Status
from pubsub.message_pb2 import DatasetDownloadComplete

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Publisher:
  """
    Initializes the Publisher client.

    Args:
      project (str): The Google Cloud project ID.
      topic (str): The Pub/Sub topic ID.
  """
  def __init__(self, project: str, topic: str):
    self.project = project
    self.topic = topic
    self.client = None
    self.topic_path = None
    try:
      # Initialize the client. This can fail if authentication is not set up.
      self.client = pubsub_v1.PublisherClient()
      self.topic_path = self.client.topic_path(project, topic)
      logging.info(f"Publisher initialized for topic: {self.topic_path}")
    except exceptions.GoogleAPICallError as e:
      logging.error(f"Failed to initialize Pub/Sub client for project '{project}': {e}")
      raise
    except Exception as e:
      logging.error(f"An unexpected error occurred during client initialization: {e}")
      raise

  def publish(self, dataset: str, destination: str) -> Status:
    msg = DatasetDownloadComplete(
      dataset=dataset,
      destination=destination,
      timestamp=datetime.datetime.now().isoformat() + 'Z'
    )
    try:
      # Data must be a bytestring
      data = msg.SerializeToString()
      # Publish the message. This returns a future.
      future = self.client.publish(self.topic_path, data)
      # Block until the message is published or the timeout is reached.
      result = future.result(timeout=30)
      logging.info(f"Published message ID {result} to {self.topic_path}")
      return Status(ok=True)
    except TimeoutError:
      logging.error(f"Publishing to {self.topic_path} timed out.")
      return Status(
        ok=False,
        message=f"Publishing to {self.topic_path} timed out."
      )
    except exceptions.NotFound:
      logging.error(f"Pub/Sub topic not found: {self.topic_path}")
      return Status(
        ok=False,
        message=f"Pub/Sub topic not found: {self.topic_path}"
      )
    except Exception as e:
      # This will catch other potential publishing errors.
      logging.error(f"An error occurred while publishing to {self.topic_path}: {e}")
      return Status(
        ok=False,
        message=f"An error occurred while publishing to {self.topic_path}: {e}"
      )