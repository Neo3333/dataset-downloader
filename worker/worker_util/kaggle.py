import os
import logging
import requests # type: ignore
import json
import time
from typing import List
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_RETRIES = 5
BACKOFF_FACTOR = 1

def get_all_dataset_files(
    owner_slug: str,
    dataset_slug: str,
    kaggle_username: str,
    kaggle_key: str,
    page_size: int=200,
  ) -> List[dict] | None:
  """
  Fetches the full list of files from a Kaggle dataset, handling token-based pagination.

  Args:
    owner_slug (str): The username of the dataset owner.
    dataset_slug (str): The name of the dataset.
    kaggle_username (str): Kaggle username in kaggle.json.
    kaggle_key (str): Kaggle key in kaggle.json.
    page_size (int): Number of files contained in one response.

  Returns:
    list: A list of dictionaries, where each dictionary represents a file.
          Returns None if there's an error.
  """
  all_files = []
  page_token = None

  if not kaggle_username or not kaggle_key:
    logging.error("Error: KAGGLE_USERNAME and KAGGLE_KEY environment variables are not set.")
    return None

  base_url = f"https://www.kaggle.com/api/v1/datasets/list/{owner_slug}/{dataset_slug}"
  params = {'pageSize': page_size, 'datasetVersionNumber': 2}
  # --- Retry logic variables ---
  retries = 0

  while True:
    # Set up parameters for the request, including the pageToken if available
    if page_token:
      params['pageToken'] = page_token
  
    retries = 0

    while retries < MAX_RETRIES:
      try:
        response = requests.get(base_url, auth=(kaggle_username, kaggle_key), params=params)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()

        # Safely get the list of files
        files_on_page = data.get('datasetFiles', [])
        if files_on_page:
          all_files.extend(files_on_page)
        
        # Break the inner loop
        page_token = data.get('nextPageToken')
        break

      except requests.exceptions.HTTPError as errh:
        # Check if the error is a 429 Too Many Requests
        if errh.response.status_code == 429:
          retries += 1
          if retries >= MAX_RETRIES:
            logging.error(f"HTTP 429: Max retries reached for page. Aborting.")
            # Return what we have so far, as this page could not be fetched.
            return all_files
            
          # Calculate sleep time with exponential backoff
          sleep_time = BACKOFF_FACTOR * (2 ** retries)
          logging.info(f"HTTP 429: Too Many Requests. Retrying in {sleep_time:.2f} seconds... (Attempt {retries}/{MAX_RETRIES})")
          time.sleep(sleep_time)
          logging.info(f"Retry on pageToken: {page_token}")
        else:
          # Handle other critical HTTP errors (404, 401, etc.)
          logging.error(f"Http Error: {errh}")
          if errh.response.status_code == 404:
            logging.error(f"Dataset not found: {owner_slug}/{dataset_slug}")
          elif errh.response.status_code == 401:
            logging.error("Authentication failed. Please check your Kaggle credentials.")
          return None # Exit completely
      except requests.exceptions.ConnectionError as errc:
        logging.error(f"Error Connecting: {errc}")
        return None
      except requests.exceptions.Timeout as errt:
        logging.error(f"Timeout Error: {errt}")
        return None
      except requests.exceptions.RequestException as err:
        logging.error(f"Something went wrong: {err}")
        return None
      except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from response: {response.text}")
        return None

    logging.info(f"Progress: {len(all_files)} files added")
    # Get the token for the next page. If it's not present or None, the loop will terminate.
    if not page_token or len(page_token) == 0:
      break
  return all_files

if __name__ == '__main__':
  # Set your Kaggle username and key as environment variables before running
  # For example, in your terminal:
  # export KAGGLE_USERNAME='your_username'
  # export KAGGLE_KEY='your_api_key'
  #
  # --- Example Usage ---
  # Replace with the owner and dataset slug you are interested in
  owner_slug = 'gpiosenka'
  dataset_slug = 'cards-image-datasetclassification'

  logging.info(f"Fetching files for dataset: {owner_slug}/{dataset_slug}")

  KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
  KAGGLE_KEY = os.getenv("KAGGLE_KEY")

  file_list = get_all_dataset_files(owner_slug, dataset_slug, KAGGLE_USERNAME, KAGGLE_KEY)

  if file_list:
    logging.info(f"Successfully retrieved {len(file_list)} files:")
    logging.info(f"Logging the first 10 files retrieved")
    for file_info in file_list[:10]:
      # Use .get() for safe dictionary access
      # The name key is 'name' and the size key is 'totalBytes'
      name = file_info.get('name', 'N/A')
      size = file_info.get('totalBytes', 'N/A')
      logging.info(f"- {name} ({size} bytes)")
  else:
    logging.error("\nCould not retrieve the file list.")