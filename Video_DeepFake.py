import cv2
import geocoder
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from keras.layers import *
from keras import *
import os
import requests
import cv2
from Audio_from_Video import video_to_audio

from audio import load_audio, classify_audio_clip
from streamlit import generate_and_save_waveform_plot
threshoold=75
filename=""
output_folder="static/Frames"
frame_interval=4
padding_factor=0.2
file="Yash_Fake_Fake.mp4"

image_dimensions = {'height': 256, 'width': 256, 'channels': 3}

face_Cap = cv2.CascadeClassifier(r"C:\Users\Vishal\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

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
##passsing face as frame
def predict_fake_or_real_frames(video_path, model, interval=10):
    print("Inside predict_fake_or_real_frames: " + video_path)
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []
    frame_counter = 0

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

    cap.release()

    if frame_predictions:
        average_prediction = np.mean(frame_predictions)
    else:
        average_prediction = 0.0

    print(average_prediction)

    return average_prediction
def entry(filename):
    # Example usage:
    video_path = filename
    print("entry: "+video_path)
    video_to_audio(video_path,"output_audio.wav")
    meso = Meso4()
    meso.load('Meso4_DF')
    print("Model done")
    average_prediction_for_video = predict_fake_or_real_frames(video_path, meso)
    print(average_prediction_for_video)
    average_prediction_for_video=(average_prediction_for_video*100)

    audio_clip = load_audio("output_audio.wav")
    result = classify_audio_clip(audio_clip)
    average_prediction_for_audio = result.item()
    average_prediction_for_audio=round(average_prediction_for_audio * 100, 2)

    generate_and_save_waveform_plot()

    if average_prediction_for_audio > 60:
        for_audio = "Fake"
    else:
        for_audio = "Real"

    if average_prediction_for_video > 75:
        print("The images are predicted to be fake.")
        for_video = "Fake"
    else:
        print("The images are predicted to be real.")
        for_video = "Real"

    if (for_video == 'Fake' and for_video == 'Fake'):
        call_or_not = 'True'
    else:
        call_or_not = 'False'

    print("Average Prediction for Video:" + str(average_prediction_for_video) + "\t" + "Categorized As:" + for_video)
    print("Average Prediction for Audio:" + str(average_prediction_for_audio) + "\t" + "Categorized As:" + for_audio)

    passobj = f'For Video -> {for_video}  ({average_prediction_for_video})  For Audio -> {for_audio} ({average_prediction_for_audio})'

    if (for_video == 'Fake' or for_audio == 'Fake'):
        telegram_send(passobj)

    if (for_video == 'Fake' or for_audio == 'Fake'):
        call()



    return for_video, for_audio, average_prediction_for_video, average_prediction_for_audio, call_or_not


###file input Face pass sigle face
def predict_real_time(model, threshold=0.85, padding_factor=0.2):
    cap = cv2.VideoCapture(file)  # Change to the appropriate camera index if needed
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_Cap.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) != 1:
            print("Warning: More or less than one face detected, skipping this frame.")
            continue

        (x, y, w, h) = faces[0]

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

        is_fake = prediction[0][0] > threshold

        # Show predictions on the original frame captured by the camera
        if is_fake:
            prediction_value = round(prediction[0][0], 2)
            display_text = f"Fake ({prediction_value})"
            cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            prediction_value = round(prediction[0][0], 2)
            display_text = f"Real ({prediction_value})"
            cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame captured by the camera with predictions
        cv2.imshow("Real-time Prediction", frame)

        # Display the extracted face region
        cv2.imshow("Extracted Face", face_roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
###file input Face pass multiple face
def predict_real_time_multiple_faces(model, threshold=0.80, padding_factor=0.2):
    cap = cv2.VideoCapture(0)  # Change to the appropriate camera index if needed
    while True:
        ret, frame = cap.read()

        if not ret:
            break

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
                print("Warning: Empty face_roi, skipping this face.")
                continue

            input_frame = preprocess_image(face_roi)
            prediction = model.predict(input_frame)

            is_fake = prediction[0][0] > threshold

            # Show predictions on the original frame captured by the camera
            if is_fake:
                prediction_value = round(prediction[0][0], 2)
                display_text = f"Fake ({prediction_value})"
                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                            cv2.LINE_AA)
            else:
                prediction_value = round(prediction[0][0], 2)
                display_text = f"Real ({prediction_value})"
                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                            cv2.LINE_AA)

            # Display the frame captured by the camera with predictions
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around the face

        # Display the frame captured by the camera with predictions
        cv2.imshow("Real-time Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



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
def call():
    key = "302de761-06ef-4301-8fce-b9e041ef21fa"
    secret = "hQck6k2sjkWvQGV9LIpmGQ=="
    fromNumber = "+447520652839"
    to = "+919022517871"
    locale = ""
    url = "https://calling.api.sinch.com/calling/v1/callouts"

    payload = {
        "method": "ttsCallout",
        "ttsCallout": {
            "cli": fromNumber,
            "destination": {
                "type": "number",
                "endpoint": to
            },
            "locale": locale,
            "text": "Hello, this is a call from Sinch. Congratulations! You made your first call."
        }
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers, auth=(key, secret))

    data = response.json()
    print(data)
def extract_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the frames count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through frames and save every frame_interval-th frame
    for frame_number in range(0, total_frames, frame_interval):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop if the frame cannot be read

        # Save the frame
        frame_filename = f"{output_folder}/frame_{frame_number:04d}.jpg"
        cv2.imwrite(frame_filename, frame)

    # Release the video capture object
    cap.release()








if __name__ == "__main__":
    meso = Meso4()
    meso.load('Meso4_DF')

    # video_path = "Sohan.mp4"
    # predict_video(meso, video_path)

    # predict_fake_or_real_frames(video_path,meso,0.75)
    # predict_real_time_multiple_faces(meso)

    #predict single face realtime
    predict_real_time(meso)


    # entry("Sohan.mp4")