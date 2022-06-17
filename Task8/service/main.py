from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from pathlib import Path
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.layers import Input, Softmax
from tensorflow.keras.models import Sequential, load_model
import numpy as np
import os

# Загрузка модели из облачного хранилища
def download_model(model_path: str):
    file_name = Path(model_path).stem
    unpack_dir = os.path.join("/tmp", file_name)

    resp = urlopen(model_path)
    zipfile = ZipFile(BytesIO(resp.read()))
    with zipfile as zip_ref:
        zip_ref.extractall(unpack_dir)
    return unpack_dir

# Формирование предсказательной модели на основе сохраненной
def get_model(path: str):
    try:
        nn_model = load_model(path)
    except (ImportError, IOError) as e:
        print("Exception: ", e)
        exit(1)
    else:
        model = Sequential()
        model.add(Input(shape=(32, 32, 3)))
        model.add(nn_model)
        model.add(Softmax())
        return model

model_dir = download_model(os.getenv('MODEL_NAME'))
model = get_model(model_dir)

# Метки датасета cifar-10
label_names = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 
               4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 
               9: "truck"}

# Основной объект FastAPI
app = FastAPI()

# Класс ответа клиенту
class CifarResponse(BaseModel):
    imagefile: str
    pred_class: str
    pred_proba: float

# Получение изображения от клиента и его предобработка
def read_imagefile(file):
    return Image.open(BytesIO(file)).resize((32, 32)).convert('RGB')

@app.get("/")
def read_healthcheck():
    return {"status": "Green", "version": "1.0"}

@app.post("/predict/")
async def predict(file: UploadFile):
    if file.content_type not in ["image/jpg", "image/png"]:
        raise HTTPException(status_code=406, detail="Only .jpg and .png files allowed")

    image = read_imagefile(await file.read())
    image = np.array(image)
    preds = model.predict(image.reshape([-1, 32, 32, 3])).reshape([-1])
    class_num = np.argmax(preds)

    resp = CifarResponse(imagefile=file.filename, 
                         pred_class=label_names[class_num], 
                         pred_proba=round(preds[class_num], 3))
    json_compatible_data = jsonable_encoder(resp)
    return JSONResponse(content=json_compatible_data)
