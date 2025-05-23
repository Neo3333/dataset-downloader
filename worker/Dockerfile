FROM python:3.11-slim
USER root
WORKDIR /worker

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY . .

# Default entrypoint: run main.py
ENTRYPOINT ["python", "main.py"]
