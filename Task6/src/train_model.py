import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Rescaling, \
    Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset

import datetime
import os
import sys
import logging
from typing import Optional
import mlflow.tensorflow


logger = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", level=logging.INFO)

__all__ = ["train_model"]

SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR_PATH = os.path.dirname(SCRIPT_PATH)
RAW_DATA_FOLDER = os.path.join(SCRIPT_DIR_PATH, "..", "data", "raw", "cifar10-dataset")
MODEL_FOLDER = os.path.join(SCRIPT_DIR_PATH, "..", "model", "cifar10-model")


def load_dataset(batch_size: int, data_path: Optional[str] = None):
    if data_path is None:
        data_path = RAW_DATA_FOLDER

    logger.info("Loading dataset from %s", data_path)
    image_size = (32, 32)
    train_dataset = image_dataset_from_directory(
        os.path.join(data_path, "train"),
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=1234,
        validation_split=0,
    )

    valid_dataset = image_dataset_from_directory(
        os.path.join(data_path, "test"),
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=1234,
        validation_split=0,
    )
    return train_dataset, valid_dataset

def set_autotuning(train_dataset, valid_dataset):
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    valid_dataset = valid_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    return train_dataset, valid_dataset

def get_model():
    model = Sequential()

    model.add(Input(shape=(32, 32, 3)))
    model.add(Rescaling(1.0 / 255))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))


    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))


    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    return model

def save_model(model, model_path: Optional[str] = None):
    if model_path is None:
        model_path = MODEL_FOLDER

    logger.info("Saving the model to: %s", model_path)
    model.save(model_path)


def train_model(data_path: Optional[str] = None, model_path: Optional[str] = None) -> None:
    if model_path is None:
        model_path = MODEL_FOLDER

    if not os.path.exists(model_path):
        mlflow.set_experiment("Cifar-10")
        mlflow.tensorflow.autolog(log_models=True)
        mlflow.start_run()

        params = {"batch_size": 128, "num_epochs": 30}

        train_dataset, valid_dataset = load_dataset(params["batch_size"], data_path)
        logger.info("Tensorflow version: %s", tf.__version__)

        class_names = train_dataset.class_names
        logger.info("class_names: %s", class_names)

        train_dataset, valid_dataset = set_autotuning(train_dataset, valid_dataset)
        model = get_model()
        model.summary()

        logger.info("Compiling the model")
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        logger.info("Training the model")
        model.fit(train_dataset,
                  validation_data=valid_dataset,
                  shuffle=True,
                  epochs=params["num_epochs"])


        save_model(model, model_path)

        mlflow.end_run()
    logging.info("Model was already trained and placed in %s", model_path)


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        train_model(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        train_model(sys.argv[1])
    else:
        train_model()