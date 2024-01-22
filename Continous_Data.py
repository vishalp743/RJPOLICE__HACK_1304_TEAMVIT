import os
import cv2
import numpy as np
from tensorflow.python.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, LeakyReLU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from keras.layers import *
from keras import *
import time
threshold=0.65
filename = ""
output_folder = "static/Frames"
frame_interval = 50
classpass=" "
xxx="xxx"

image_dimensions = {'height': 256, 'width': 256, 'channels': 3}

class Classifier:
    def __init__(self):
        self.model = 0

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)

class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()

        self.model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

    def init_model(self):
        x = Input(shape=(image_dimensions['height'], image_dimensions['width'], image_dimensions['channels']))

        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)


def preprocess_image(image):
    input_size = (256, 256)
    resized_image = cv2.resize(image, input_size)
    normalized_image = resized_image / 255.0
    input_image = normalized_image.reshape((1,) + normalized_image.shape)
    return input_image

def get_latest_image(folder_path):
    images = [f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")]
    if images:
        return os.path.join(folder_path, max(images, key=os.path.getctime))
    else:
        return None

import requests

def predict_fake_or_real_images(folder_path, model, threshold=0.65):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            input_image = preprocess_image(cv2.imread(image_path))
            prediction = model.predict(input_image)
            classpass = "Fake" if prediction[0][0] > threshold else "Real"
            print(image_path)
            print(classpass)
            print(prediction[0][0])
        else:
            print("Exit")
            continue
    return classpass



def entryimgfolder1(folder_path, timeout_sec=60):
    meso = Meso4()
    meso.load('Meso4_DF')

    predictions = predict_fake_or_real_images(folder_path, meso, threshold=0.65)
    print(predictions)
    return predictions



