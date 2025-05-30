FROM python:3.11-slim
WORKDIR /app

COPY shared/ /app/shared
COPY model_service/ /app/model_service/

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
RUN pip install -r model_service/requirements.txt
ENV PYTHONPATH=/app

CMD ["python", "model_service/inference.py"]
