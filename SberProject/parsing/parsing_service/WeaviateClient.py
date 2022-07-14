from datetime import datetime, timezone
import time
import weaviate
import hashlib
import logging
import threading
import os


class SingletonSchemaCreator:
    _instance = None
    _lock = threading.Lock()

    def __check_and_create_schema(self, weaviate_client: weaviate.Client):
        schema = {
            "classes": [{
                "class": "SberEntity",
                "vectorizer": "text2vec-transformers",
                "properties": [{
                    "name": "title",
                    "dataType": ["text"],
                }, {
                    "name": "uri",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": "true",
                        }
                    },
                }, {
                    "name": "content",
                    "dataType": ["text"],
                }, {
                    "name": "datetime",
                    "dataType": ["date"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": "true",
                        }
                    },
                }]
            }]
        }

        logger = logging.getLogger(__name__)
        if not weaviate_client.schema.contains(schema):
            weaviate_client.schema.delete_all()
            weaviate_client.schema.create(schema)
            logger.warning("Schema was recreated! Old data were removed!")
        else:
            logger.info("Schema is already exists!")

    def __new__(cls, weaviate_client: weaviate.Client):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(SingletonSchemaCreator, cls).__new__(cls)
                cls._instance.__check_and_create_schema(weaviate_client)
        return cls._instance


class WeaviateClient:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = self.__wait_for_weaviate_ready(10, 30)
        SingletonSchemaCreator(self.client)

    def __wait_for_weaviate_ready(self, time_to_wait: int = 10, num_attempts: int = 30):
        weaviate_host = os.getenv('WEAVIATE_HOST', default="localhost")
        weaviate_port = os.getenv('WEAVIATE_PORT', default="8080")
        self.logger.info(f"Value of env variable 'WEAVIATE_HOST': {weaviate_host}")
        self.logger.info(f"Value of env variable 'WEAVIATE_PORT': {weaviate_port}")
        self.logger.info("Selenium parsing worker started.")

        for attempt in range(num_attempts):
            try:
                self.logger.info(f"Attempt {attempt + 1} to connect to http://{weaviate_host}:{weaviate_port}...")
                return weaviate.Client(f"http://{weaviate_host}:{weaviate_port}",
                                       timeout_config=(20, 120))
            except:
                self.logger.warning(f"Attempt was failed! Connection refused!")
                time.sleep(time_to_wait)
        raise ConnectionRefusedError("Connection refused!")

    def __create_data_object(self, data_object: dict[str, str], uuid: str):
        try:
            self.client.data_object.create(data_object=data_object,
                                           class_name='SberEntity',
                                           uuid=uuid)
            self.logger.info(f"<{data_object['title']}> was created!")
        except Exception as e:
            self.logger.warning(f"EXCEPTION DURING CREATING <{data_object['title']}>: ", e)

    def __update_data_object(self, data_object: dict[str, str], uuid: str):
        try:
            self.client.data_object.update(data_object=data_object,
                                           class_name='SberEntity',
                                           uuid=uuid)
            self.logger.info(f"<{data_object['title']}> was updated!")
        except Exception as e:
            self.logger.warning(f"EXCEPTION DURING UPDATING <{data_object['title']}>: ", e)

    def post_data(self, data: dict):
        local_time = datetime.now(timezone.utc).astimezone()
        data_object = {"title": data["title"],
                       "uri": data["uri"],
                       "content": data["content"],
                       "datetime": local_time.isoformat(),
                       }
        uuid = hashlib.md5(data_object["title"].encode()).hexdigest()

        if self.client.data_object.exists(uuid):
            self.__update_data_object(data_object, uuid)
        else:
            self.__create_data_object(data_object, uuid)
