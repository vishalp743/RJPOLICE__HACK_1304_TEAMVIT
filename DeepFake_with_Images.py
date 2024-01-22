import cv2
import geocoder
import numpy as np
from tensorflow.python.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, LeakyReLU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from keras.layers import *
from keras import *
import os
from audio import load_audio,classify_audio_clip
import requests

from streamlit import generate_and_save_waveform_plot

padding_factor=0.2
face_Cap = cv2.CascadeClassifier(r"C:\Users\Vishal\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

from Audio_from_Video import video_to_audio

filename=""
output_folder="Down_Frames"
frame_interval=50
folder_path='static/Frames'

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
def predict_fake_or_real_frames1(video_path, model, interval=10, padding_factor=0.2):
    print("Inside predict_fake_or_real_frames: " + video_path)
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []
    frame_counter = 0
    highest_prediction = 0.0
    highest_frame = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_counter += 1

        if frame_counter % interval != 0:
            continue  # Skip frames based on the interval

        # Face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_Cap.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            # Add padding around the detected face
            padding_x = int(w * padding_factor)
            padding_y = int(h * padding_factor)

            face_roi = frame[max(0, y - padding_y):min(frame.shape[0], y + h + padding_y),
                       max(0, x - padding_x):min(frame.shape[1], x + w + padding_x)]

            if face_roi.size == 0:
                print("Warning: Empty face_roi, skipping this frame.")
                continue

            input_frame = preprocess_image(face_roi)
            prediction = model.predict(input_frame)

            frame_predictions.append(prediction[0][0])

            # Check if the current frame has a higher prediction value
            if prediction[0][0] > highest_prediction:
                highest_prediction = prediction[0][0]
                highest_frame = frame.copy()

    cap.release()

    if frame_predictions:
        average_prediction = np.mean(frame_predictions)
    else:
        average_prediction = 0.0

    print("Average Prediction:", average_prediction)

    if highest_frame is not None:
        output_path = "best_frame.jpg"
        cv2.imwrite(output_path, highest_frame)
        print("Highest frame saved:", output_path)

    return average_prediction
def telegram_send(passobj):
    print("entered in telegram")
    location = ""
    g = geocoder.ip('me')
    if g.latlng is not None:
        location = g.latlng

    base_url = "https://api.telegram.org/bot6416572645:AAFQQhRiAosOOHZDFgH3H7hoWVyl5J7aE1Y/sendPhoto"

    image_path = "C:/Users/Vishal/OneDrive/Desktop/DeepFake/best_frame.jpg"
    while not os.path.exists(image_path):
        pass  # Wait until the file exists

    my_file = open(image_path, "rb")

    parameters = {
        "chat_id" : "998635769",
        "caption" : f"{passobj}"
    }


    files = {
        "photo" : my_file
    }

    resp = requests.get(base_url, data = parameters, files=files)
    print(resp.text)
    print("leaved telegram")
def entryimgfolder(File_Path):
    call_or_not='False'

    meso = Meso4()
    meso.load('Meso4_DF')

    average_prediction_for_video = predict_fake_or_real_frames1(File_Path, meso)
    average_prediction_for_video = (average_prediction_for_video * 100)-11

    audio_clip = load_audio("output_audio.wav")
    result = classify_audio_clip(audio_clip)
    average_prediction_for_audio = result.item()
    average_prediction_for_audio = round(average_prediction_for_audio * 100, 2)+8

    generate_and_save_waveform_plot()

    if average_prediction_for_audio > 60:
        for_audio = "Fake"
    else:
        for_audio = "Real"

    if average_prediction_for_video > 70:
        print("The images are predicted to be fake.")
        for_video = "Fake"
    else:
        print("The images are predicted to be real.")
        for_video = "Real"

    if (for_video == 'Fake' or for_video == 'Fake'):
        call_or_not = 'True'
    else:
        call_or_not = 'False'

    print("Average Prediction for Video:" + str(average_prediction_for_video) + "\t" + "Categorized As:" + for_video)
    print("Average Prediction for Audio:" + str(average_prediction_for_audio) + "\t" + "Categorized As:" + for_audio)

    passobj = f'For Video -> {for_video}  ({average_prediction_for_video})  For Audio -> {for_audio} ({average_prediction_for_audio})'

    if (for_video == 'Fake' or for_audio == 'Fake'):
        telegram_send(passobj)

    return for_video,for_audio,average_prediction_for_video,average_prediction_for_audio,call_or_not


