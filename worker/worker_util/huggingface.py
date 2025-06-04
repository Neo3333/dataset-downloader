import os
import logging
import requests # type: ignore

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def check_datasets_server_parquet_status(
    repo_id: str, token: str = None, show_files: bool=False) -> dict:
  """
  Checks the Parquet conversion status of a Hugging Face dataset using the
  datasets-server.huggingface.co/parquet endpoint.

  Args:
    repo_id (str): The ID of the dataset repository (e.g., "squad", "glue").
    token (str, optional): A Hugging Face API token if dealing with private datasets.
                           (May not be respected by this specific public endpoint,
                           but good practice if generalising). Defaults to None.
    show_files (bool): Whether to include the full file list in the "configuration"
                          section.

  Returns:
    dict: A dictionary containing:
      - "available" (bool): True if Parquet info is available from the endpoint.
      - "is_partial" (bool | None): The value of the 'partial' field from the
                                    response. None if not found or in case of error.
      - "configurations" (list | None): List of dataset configurations with
                                       Parquet file details. None if not found
                                       or in case of error.
      - "error_message" (str | None): Error message if any occurred.
      - "http_status_code" (int | None): HTTP status code of the API response.
  """
  status = {
    "available": False,
    "is_partial": None,
    "configurations": None,
    "error_message": None,
    "http_status_code": None,
  }

  headers = {}
  if token:
    # Note: datasets-server is typically a public-facing service,
    # but including token handling for completeness if it ever applies.
    headers["Authorization"] = f"Bearer {token}"

  # Construct the URL
  # The datasets-server endpoint is typically for public datasets.
  # For private datasets, this specific endpoint might not work or might require
  # different authentication than the main Hub API.
  endpoint_url = f"https://datasets-server.huggingface.co/parquet?dataset={repo_id}"

  try:
    response = requests.get(endpoint_url, headers=headers, timeout=15) # Added timeout
    status["http_status_code"] = response.status_code

    # Check for common HTTP errors that indicate non-availability or issues
    if response.status_code == 404:
      status["error_message"] = (
        f"Dataset '{repo_id}' not found or no Parquet conversion "
        f"available via datasets-server (404 error)."
      )
      status["available"] = False
      return status
    elif response.status_code == 500: # Internal server error
      status["error_message"] = (
        f"datasets-server internal error for '{repo_id}' (500 error). "
        "The server might be temporarily unavailable or the dataset problematic."
      )
      status["available"] = False
      return status
    
    response.raise_for_status()  # Raise an exception for other HTTP errors (4xx, 5xx)

    data = response.json()

    if "error" in data: # Check for explicit error messages in the JSON response
      status["error_message"] = f"API returned an error: {data['error']}"
      status["available"] = False
      return status
    
    if "configurations" in data and isinstance(data["configurations"], list) and data["configurations"]:
      status["available"] = True
      if show_files:
        status["configurations"] = data["configurations"]
      
      if "partial" in data: # Check for a global 'partial' flag
        status["is_partial"] = bool(data["partial"])
      else:
        any_config_is_partial = False
        all_configs_have_partial_flag = True
        for config_info in data["configurations"]:
          if "partial" in config_info and isinstance(config_info["partial"], bool):
            if config_info["partial"]:
              any_config_is_partial = True
              break # If one is partial, we can stop
          else:
            all_configs_have_partial_flag = False 
        
        if all_configs_have_partial_flag or any_config_is_partial :
          status["is_partial"] = any_config_is_partial
        else:
          if status["available"]: 
            logger.info(f"Warning: 'partial' field not consistently found in configurations for {repo_id}. Partial status undetermined.")
          status["is_partial"] = None


      has_actual_files = False
      for config_info in status["configurations"]:
        if "parquet_files" in config_info and isinstance(config_info["parquet_files"], list) and config_info["parquet_files"]:
          has_actual_files = True
          break
      if not has_actual_files:
        status["available"] = False 
        status["error_message"] = (
          "Parquet configurations found, but no Parquet files listed within them."
        )

    elif "parquet_files" in data and isinstance(data["parquet_files"], list): # Simpler structure
      status["available"] = True if data["parquet_files"] else False
      if show_files:
        status["configurations"] = [
          {"dataset": repo_id, "config": repo_id, "parquet_files": data["parquet_files"]}
        ] 
      if "partial" in data and isinstance(data["partial"], bool):
        status["is_partial"] = data["partial"]
      else:
        status["is_partial"] = None 
        if status["available"]:
          logger.info(f"Warning: 'partial' field not found in response for {repo_id} (simple structure). Partial status undetermined.")
      if not data["parquet_files"] and status["available"]:
        status["available"] = False
        status["error_message"] = "Parquet files list is empty."

    else: 
      status["available"] = False
      status["error_message"] = (
        "No Parquet configurations or files found in the response, or unexpected response structure."
      )

  except requests.exceptions.HTTPError as e:
    status["error_message"] = f"HTTP error during API request: {e}"
    status["available"] = False 
  except requests.exceptions.RequestException as e:
    status["error_message"] = f"API request failed (e.g., network issue): {e}"
  except ValueError as e:  # JSON decoding error
    status["error_message"] = f"Failed to parse API JSON response: {e}"
  except Exception as e:
    status["error_message"] = f"An unexpected error occurred: {e}"

  return status

# --- Example Usage ---
if __name__ == "__main__":
  resp = check_datasets_server_parquet_status(
    repo_id='bigcode/the-stack-v2-train-full-ids',
    token=os.getenv("HF_HUB_TOKEN")
  )
  print(resp)