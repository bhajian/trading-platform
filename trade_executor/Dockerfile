FROM python:3.11-slim
WORKDIR /app

COPY shared/ /app/shared
COPY trade_executor/ /app/trade_executor/

RUN pip install --no-cache-dir -r trade_executor/requirements.txt
ENV PYTHONPATH=/app

CMD ["python", "trade_executor/executor.py"]
