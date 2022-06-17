FROM python:3.10

WORKDIR /usr/src/service

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY service ./service

RUN useradd --user-group --shell /bin/false uvicorn
USER uvicorn
WORKDIR /usr/src/service

EXPOSE 8080

ENV MODEL_NAME "cifar10-model"

ENTRYPOINT ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080"]
