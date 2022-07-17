FROM python:3.10

WORKDIR /usr/src/server

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY server ./server

RUN useradd --user-group --shell /bin/false uvicorn && chown -R uvicorn:uvicorn .
USER uvicorn

EXPOSE 8080

ENTRYPOINT ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
