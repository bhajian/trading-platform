FROM python:3.11-slim
WORKDIR /app

COPY shared/ /app/shared
COPY data_loader/ /app/data_loader/

RUN pip install --no-cache-dir -r data_loader/requirements.txt
ENV PYTHONPATH=/app

CMD ["python", "data_loader/loader.py"]
