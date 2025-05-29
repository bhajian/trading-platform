# model_service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Torch needs a few build utils; keep image small
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "model_service.inference"]
