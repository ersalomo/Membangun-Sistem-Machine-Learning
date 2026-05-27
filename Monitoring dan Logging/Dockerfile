FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY 3.prometheus_exporter.py /app/
COPY model.joblib /app/

EXPOSE 8000

ENV PORT=8000
ENV MODEL_PATH=model.joblib

CMD ["python", "3.prometheus_exporter.py"]
