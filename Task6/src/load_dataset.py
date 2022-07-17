import os
import sys
from typing import Optional
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", level=logging.INFO)

__all__ = ["load_dataset"]

URL = "https://storage.yandexcloud.net/kpd-public-bucket/cifar10-dataset.zip"
SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR_PATH = os.path.dirname(SCRIPT_PATH)
RAW_DATA_FOLDER = os.path.join(SCRIPT_DIR_PATH, "..", "data", "raw", "cifar10-dataset")


def download_model(data_path: Optional[str] = None) -> None:
    if data_path is None:
        data_path = RAW_DATA_FOLDER

    if not os.path.exists(data_path):
        logger.info("Downloading dataset from %s", URL)
        resp = urlopen(URL)
        zipfile = ZipFile(BytesIO(resp.read()))
        with zipfile as zip_ref:
            zip_ref.extractall(data_path)
    logging.info("Dataset is already downloaded in %s", data_path)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        download_model(sys.argv[1])
    else:
        download_model()
