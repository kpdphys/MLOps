FROM python:3.10.1

WORKDIR /usr/src/search

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY search_service ./search_service

RUN useradd --user-group --shell /bin/bash uvicorn
USER uvicorn

ENTRYPOINT ["uvicorn", "search_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
