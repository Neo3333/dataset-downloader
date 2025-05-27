FROM python:3.11-slim
USER root
WORKDIR /worker

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY . ./

# Add an entrypoint script to login to HF and then run
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

# Use the entrypoint to authenticate and then launch the worker
ENTRYPOINT ["./entrypoint.sh"]
