FROM python:3.11-slim
WORKDIR /app

COPY shared/ /app/shared
COPY decision_service/ /app/decision_service/

RUN pip install --no-cache-dir -r decision_service/requirements.txt
ENV PYTHONPATH=/app

CMD ["python", "decision_service/decision_service.py"]
