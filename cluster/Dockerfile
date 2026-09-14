ARG PYTHON_IMAGE
FROM ${PYTHON_IMAGE}
WORKDIR /app
COPY requirements/services.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /app/requirements.lock
ARG SERVICE
COPY reference/stage3_scale/infosys-benchmark/${SERVICE}/app.py /app/app.py
EXPOSE 8080
CMD ["python", "-u", "app.py"]
