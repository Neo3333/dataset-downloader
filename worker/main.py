import argparse

from flask import Flask, request, jsonify # type: ignore
from .dataset_downloader import download_dataset
from .config import FLASK_HOST, FLASK_PORT

app = Flask(__name__)

@app.route("/download", methods=["POST"])
def download_endpoint():
  """HTTP endpoint to trigger dataset download via JSON payload"""
  data = request.get_json(force=True)
  repo_id = data.get("dataset")
  config = data.get("config")
  split = data.get("split")
  dest = data.get("destination") or None

  if not repo_id:
    return jsonify({"error": "Missing 'dataset' parameter"}), 400

  try:
    download_dataset(repo_id, config=config, split=split, dest=dest)
    return jsonify({"status": "success", "dataset": repo_id}), 200
  except Exception as e:
    print(f"Error: {e}")
    return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="HuggingFace Dataset Download Worker"
  )
  parser.add_argument(
      "--dataset", required=True,
      help="HuggingFace dataset repo ID (e.g. user/dataset)"
  )
  parser.add_argument(
      "--config", help="Dataset config name (if applicable)"
  )
  parser.add_argument(
      "--split", help="Dataset split (e.g. train, test)"
  )
  parser.add_argument(
      "--destination", help="Destination directory to save dataset"
  )
  args = parser.parse_args()

  download_dataset(
    repo_id=args.dataset,
    config=args.config,
    split=args.split,
    dest=args.destination
  )
else:
  # If imported, run Flask app
  app.run(host=FLASK_HOST, port=FLASK_PORT)