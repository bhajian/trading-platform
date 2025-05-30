FROM python:3.11-slim
WORKDIR /app

COPY shared/ /app/shared
COPY data_retainer/ /app/data_retainer/

RUN pip install --no-cache-dir -r data_retainer/requirements.txt
ENV PYTHONPATH=/app

CMD ["python", "data_retainer/retainer.py"]
