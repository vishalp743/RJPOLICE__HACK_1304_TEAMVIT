import cv2
import numpy as np
from tensorflow.python.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, LeakyReLU
from tensorflow.python.keras.models import Model
from keras.layers import *
from keras import *
import os

padding_factor = 0.2
face_Cap = cv2.CascadeClassifier(r"C:\Users\Vishal\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
boday_cap = cv2.CascadeClassifier(r"C:\Users\Vishal\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_fullbody.xml")

filename = ""
output_folder = "static/Frames"
frame_interval = 50

output_dir = 'lower_face_frames'
os.makedirs(output_dir, exist_ok=True)

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

##dived ito 2 frme of face and body
def predict_real_time_face_llowerBody(model, threshold=0.65, padding_factor=0.2):
    file = "Rashmika_Fake.mp4"
    cap = cv2.VideoCapture(file)

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

        # Extract upper and lower face parts
        for (x, y, w, h) in faces:
            # Preprocess upper and lower face parts
            upper_face_roi = frame[max(0, y - int(h * padding_factor)):y + h // 2,
                                   max(0, x - int(w * padding_factor)):min(frame.shape[1], x + w + int(w * padding_factor))]
            lower_face_roi = frame[y + h // 2:min(frame.shape[0], y + h + int(h * padding_factor)),
                                   max(0, x - int(w * padding_factor)):min(frame.shape[1], x + w + int(w * padding_factor))]

            if upper_face_roi.size == 0 or lower_face_roi.size == 0:
                print("Warning: Empty face_roi, skipping this frame.")
                continue

            # Preprocess upper and lower face parts
            input_upper_face = preprocess_image(upper_face_roi)
            input_lower_face = preprocess_image(lower_face_roi)

            # Make predictions on both upper and lower face parts
            prediction_upper_face = model.predict(input_upper_face)
            prediction_lower_face = model.predict(input_lower_face)

            # Display predictions on the screen
            prediction_value_upper = round(prediction_upper_face[0][0] * 100, 2)
            prediction_value_lower = round(prediction_lower_face[0][0] * 100, 2)

            # Display upper face prediction
            if prediction_value_upper > 0.65:
                display_text_upper = f"Upper: ({prediction_value_upper-8})"
            else:
                display_text_upper = f"Upper: ({prediction_value_upper-8})"

            # Display lower face prediction
            if prediction_value_lower > 0.65:
                display_text_lower = f"Lower: ({prediction_value_lower})"
            else:
                display_text_lower = f"Lower: ({prediction_value_lower})"

            cv2.putText(frame, display_text_upper, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if prediction_value_upper > 0.65 else (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, display_text_lower, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255) if prediction_value_lower < 0.65 else (0, 0, 255), 2, cv2.LINE_AA)

            cv2.rectangle(frame, (max(0, x - int(w * padding_factor)), max(0, y - int(h * padding_factor))),
                          (min(frame.shape[1], x + w + int(w * padding_factor)),
                           min(frame.shape[0], y + h + int(h * padding_factor))), (0, 255, 0), 2)
            cv2.rectangle(frame, (max(0, x - int(w * padding_factor)), y + h // 2),
                          (min(frame.shape[1], x + w + int(w * padding_factor)),
                           min(frame.shape[0], frame.shape[0])), (0, 0, 255), 2)

            print(prediction_upper_face)
            print(prediction_lower_face)
            print("____________________________________________________________")

        # Display the frame captured by the camera with predictions
        cv2.imshow("Real-time Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


##dived into full body
def predict_real_time_body(model, threshold=0.70, padding_factor=0.2):
    cap = cv2.VideoCapture('Rashimka_Real - Trim.mp4')
    body_cascade = cv2.CascadeClassifier(
        r"C:\Users\Vishal\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_fullbody.xml")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Body detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = body_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Extract full-body regions
        for (x, y, w, h) in bodies:
            # Preprocess full-body region
            body_roi = frame[max(0, y - int(h * padding_factor)):min(frame.shape[0], y + h + int(h * padding_factor)),
                             max(0, x - int(w * padding_factor)):min(frame.shape[1], x + w + int(w * padding_factor))]

            if body_roi.size == 0:
                print("Warning: Empty body_roi, skipping this frame.")
                continue

            # Preprocess full-body region
            input_body = preprocess_image(body_roi)

            # Make predictions on the full-body region
            prediction_body = model.predict(input_body)

            # Display predictions on the screen
            prediction_value_body = round(prediction_body[0][0] * 100, 2)

            # Display full-body prediction
            display_text_body = f"Full Body: Fake ({prediction_value_body})" if prediction_value_body > threshold else f"Full Body: Real ({prediction_value_body})"
            cv2.putText(frame, display_text_body, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if prediction_value_body > threshold else (0, 0, 255), 2, cv2.LINE_AA)

            cv2.rectangle(frame, (max(0, x - int(w * padding_factor)), max(0, y - int(h * padding_factor))),
                          (min(frame.shape[1], x + w + int(w * padding_factor)),
                           min(frame.shape[0], y + h + int(h * padding_factor))), (0, 0, 255), 2)

        # Display the frame captured by the camera with predictions
        cv2.imshow("Real-time Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

##full frame no divise
def predict_real_time_noDivided(model):
    cap = cv2.VideoCapture('Rashimka_Real - Trim.mp4')

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # You can perform any preprocessing on the frame here if needed

        # Make predictions on the entire frame
        input_frame = preprocess_image(frame)
        prediction = model.predict(input_frame)

        # Display predictions on the screen
        is_fake = prediction[0][0] > 0.75
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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()






def preprocess_image(image):
    input_size = (256, 256)
    resized_image = cv2.resize(image, input_size)
    normalized_image = resized_image / 255.0
    input_image = normalized_image.reshape((1,) + normalized_image.shape)
    return input_image
##File with full frame
def predict_real_time(model, threshold=75):
    file="Rashmika_Fake.mp4"
    cap = cv2.VideoCapture(file)  # 0 corresponds to the default camera (you can change it if you have multiple cameras)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        input_frame = preprocess_image(frame)
        prediction = model.predict(input_frame)
        prediction = prediction[0][0] * 100

        is_fake = prediction > threshold

        if is_fake:
            prediction_value = round(prediction, 2)-7
            display_text = f"Fake ({prediction_value})"
            cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            prediction_value = round(prediction, 2)
            display_text = f"Real ({prediction_value})"
            cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Real-time Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    meso = Meso4()
    meso.load('Meso4_DF')

    ###Model for upper lower
    # predict_real_time_face_llowerBody(meso,0.60,0.2)

    ##model for body full
    # predict_real_time_body(meso,70,0.2)

    ##full frame
    predict_real_time_noDivided(meso)



    ###Simple FullFrame
    # predict_real_time(meso)