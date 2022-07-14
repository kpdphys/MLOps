import os
import time
import logging
import weaviate
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", level=logging.INFO)


def __wait_for_weaviate_ready(time_to_wait: int = 10,
                              num_attempts: int = 20):
    weaviate_host = os.getenv('WEAVIATE_HOST', default="localhost")
    weaviate_port = os.getenv('WEAVIATE_PORT', default="8080")
    logger.info(f"Value of env variable 'WEAVIATE_HOST': {weaviate_host}")
    logger.info(f"Value of env variable 'WEAVIATE_PORT': {weaviate_port}")
    logger.info("Selenium parsing worker started.")

    for attempt in range(num_attempts):
        try:
            logger.info(f"Attempt {attempt + 1} to connect to http://{weaviate_host}:{weaviate_port}...")
            return weaviate.Client(f"http://{weaviate_host}:{weaviate_port}", timeout_config=(20, 120))
        except:
            logger.warning(f"Attempt was failed! Connection error!")
            time.sleep(time_to_wait)
    raise ConnectionRefusedError("Connection error!")


client = __wait_for_weaviate_ready(10, 20)
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_healthcheck():
    print("Healthcheck request.")
    return {"status": "Green", "version": "1.0"}


@app.get("/search/")
async def search(q: str):
    near_text = {"concepts": [q], "certainty": 0.7}
    answer = client.query.get("SberEntity",
                              ["title", "uri"]
                              ).with_near_text(near_text).with_limit(5).do()
    return json.dumps({"search": answer["data"]["Get"]["SberEntity"]},
                      ensure_ascii=False)
