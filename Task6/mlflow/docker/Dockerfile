FROM python:3.7-slim-buster

RUN pip install mlflow boto3 pymysql

RUN useradd --user-group --shell /bin/bash mlflow
USER mlflow
