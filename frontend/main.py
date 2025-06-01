import logging
import re
from distutils.util import strtobool

from flask import Flask, request, jsonify # type: ignore
from job_trigger import trigger_download_job
from config import FLASK_HOST, FLASK_PORT

app = Flask(__name__)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def is_valid_dataset(dataset: str) -> bool:
  return bool(re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", dataset))

def is_valid_suffix_format(suffix: str) -> bool:
  if suffix.startswith("/") or suffix.endswith("/"):
    return False
  parts = suffix.split("/")
  return all(part and part not in (".", "..") for part in parts)

@app.route('/enqueue', methods=['POST'])
def enqueue():
  data = request.get_json(force=True)
  dataset = data.get('dataset')
  source = data.get('source')
  dest_suffix = data.get('dest_suffix', '')

  if not isinstance(dataset, str) or not dataset.strip():
    return jsonify({'error': "'dataset' must be a non-empty string"}), 400

  if not is_valid_dataset(dataset):
    return jsonify({'error': f"Non valid 'dataset' field {dataset}"}), 400

  if not source or source not in ['kaggle', 'huggingface']:
    return jsonify({'error': f"Non valid 'source' field {source}"}), 400

  if dest_suffix and not is_valid_suffix_format(dest_suffix):
    return jsonify({'error': f"Invalid destination suffix: {dest_suffix}"}), 400

  if not dest_suffix:
    dest_suffix = dataset

  operation_id, status = trigger_download_job(
    dataset=dataset,
    source=source,
    dest_suffix=dest_suffix
  )

  if not status.is_ok():
    return jsonify({'error': status.message, 'code': status.code}), 500

  return jsonify({
    'status': 'enqueued',
    'execution': operation_id
  }), 202

if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT)