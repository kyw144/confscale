FROM python:3.12-slim

RUN pip install flask prometheus-flask-exporter requests redis

WORKDIR /app
COPY app.py .

EXPOSE 8080
CMD ["python", "-u", "app.py"]
