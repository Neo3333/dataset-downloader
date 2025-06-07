FROM python:3.11-slim
USER root
WORKDIR /worker-job

# Install dependencies
COPY ./worker/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code

COPY ./worker/ /worker-job/worker/
COPY ./util/ /worker-job/util/
COPY ./gcs/ /worker-job/gcs/

# Add an entrypoint script to login to HF and then run
COPY ./worker/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

ENV PYTHONPATH="${PYTHONPATH}:/worker-job"

# Use the entrypoint to authenticate and then launch the worker
ENTRYPOINT ["./entrypoint.sh"]
