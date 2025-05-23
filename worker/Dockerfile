FROM python:3.11-slim
USER root
WORKDIR /worker

# Install dependencies
COPY worker/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY worker/ ./

# Default entrypoint: run main.py
ENTRYPOINT ["python", "main.py"]
