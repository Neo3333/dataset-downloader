FROM python:3.11-slim
USER root
WORKDIR /app

# Install dependencies
COPY ./frontend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY ./frontend/ /app/frontend
COPY ./util/ /app/util

# Expose port
EXPOSE 8080
ENV PORT 8080

# Start Gunicorn
CMD ["gunicorn", "frontend.main:app", "--bind", ":8080", "--workers", "1", "--threads", "8", "--timeout", "0"]