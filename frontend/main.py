import logging
from flask import Flask, request, jsonify # type: ignore

from job_trigger import trigger_download_job
from config import FLASK_HOST, FLASK_PORT

app = Flask(__name__)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/enqueue', methods=['POST'])
def enqueue():
  data = request.get_json(force=True)
  dataset = data.get('dataset')
  destination= data.get('destination', '/mnt/filestore')

  if not isinstance(dataset, str) or not dataset.strip():
    return jsonify({'error': "'dataset' must be a non-empty string"}), 400

  execution_name, status = trigger_download_job(dataset=dataset, destination=destination)

  if not status.is_ok():
    return jsonify({'error': status.message, 'code': status.code}), 500

  return jsonify({
    'status': 'enqueued',
    'execution': execution_name
  }), 202

if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT)