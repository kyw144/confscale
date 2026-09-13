FROM python:3.12-slim

RUN pip install flask prometheus-flask-exporter gunicorn

WORKDIR /app
COPY app.py .

EXPOSE 8080
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
