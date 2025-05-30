FROM python:3.11-slim
WORKDIR /app

COPY shared/ /app/shared
COPY trade_manager/ /app/trade_manager/

RUN pip install --no-cache-dir -r trade_manager/requirements.txt
ENV PYTHONPATH=/app

CMD ["python", "trade_manager/manager.py"]
