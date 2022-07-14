FROM python:3.10

WORKDIR /usr/src/metrics

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY metrics_service ./metrics_service
COPY __init__.py ./__init__.py

RUN useradd --user-group --shell /bin/bash uvicorn
USER uvicorn

ENTRYPOINT ["uvicorn", "metrics_service.server.service:app", "--host", "0.0.0.0", "--port", "8080"]
