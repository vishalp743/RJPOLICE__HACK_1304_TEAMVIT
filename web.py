import io

import librosa
from flask import Flask, render_template, url_for, request, jsonify, Response
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas

from Video_DeepFake import entry, extract_frames, call, Meso4
import os
from DeepFake_with_Images import entryimgfolder
from Audio_from_Video import  video_to_audio
import cv2
import numpy as np
face_Cap = cv2.CascadeClassifier(r"C:\Users\Vishal\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
from Continous_Data import  entryimgfolder1
app = Flask(__name__)

classpass=" "
selected_images=['frame_0.png', 'frame_0.png', 'frame_0.png', 'frame_0.png', 'frame_0.png','frame_0.png']
@app.route('/')
@app.route('/home')
def home():
    print('Inside Home')
    frames_directory = "static/Frames"
    image_files = [file for file in os.listdir(frames_directory) if file.endswith(".png")]
    display_images = image_files[:6]
    return render_template('index.html', images=display_images)
@app.route('/aboutpage')
def aboutpage():
    return render_template('about.html')
@app.route('/oneClick')
def oneClick():
    return render_template('oneclick.html')


####Video Pass WIhtout face whole
@app.route('/success', methods = ['POST'])
def success():
    print('Inside sucess')
    if request.method == 'POST':
        frames_directory = "static/Frames"
        f = request.files['file']
        f.save(f.filename)
        extract_frames(f.filename)
        image_files = [file for file in os.listdir(frames_directory) if file.endswith(".jpg")]
        display_images = image_files[:6]
        print(display_images)
        for_video,for_audio,average_prediction_for_video,average_prediction_for_audio,call_or_not = entry(f.filename)
        return render_template("oneclick.html", for_video=for_video, for_audio=for_audio,
                               average_prediction_for_video=average_prediction_for_video,
                               average_prediction_for_audio=average_prediction_for_audio, call_or_not=call_or_not,images=display_images)

##FACE IMAGE PASSS
@app.route('/entryimgfolder', methods = ['POST'])
def hello2():
    print('Inside entryimgfolder')
    file_name = request.cookies.get('fileName', '')
    video_to_audio(file_name,"output_audio.wav")
    images_string = request.cookies.get('Images', '')
    selected_images = images_string.split(',')
    File_Path=file_name
    for_video,for_audio,average_prediction_for_video,average_prediction_for_audio,call_or_not = entryimgfolder(File_Path)
    if for_video == 'Fake' or for_audio == 'Fake':
        call()
    scroll="Yes"
    return render_template("index.html", for_video=for_video,for_audio=for_audio,average_prediction_for_video=average_prediction_for_video,average_prediction_for_audio=average_prediction_for_audio,call_or_not=call_or_not,scroll=scroll)



###rendering
@app.route('/temp')
def temp():
    print('Inside temp')
    return render_template('temp.html')
####CALLL
@app.route('/call', methods = ['POST'])
def hello():
    print('Inside call')
    call()
    return render_template("index.html", res="Call Done",images=selected_images)


##LIVE FACE DETECTION
# @app.route('/entryimgfolder1', methods = ['POST'])
# def hello3():
#     print('Inside entryimgfolder1')
#     file_name = request.cookies.get('fileName', '')
#     video_to_audio(file_name, "output_audio.mp3")
#     images_string = request.cookies.get('Images', '')
#     selected_images = images_string.split(',')
#     for_video, for_audio, average_prediction_for_video, average_prediction_for_audio, call_or_not = entryimgfolder(
#         "static/Frames")
#     return render_template("index.html", for_video=for_video, for_audio=for_audio,
#                            average_prediction_for_video=average_prediction_for_video,
#                            average_prediction_for_audio=average_prediction_for_audio, call_or_not=call_or_not)


###TEMP ROUTES
# @app.route('/result', methods=['POST', 'GET'])
# def result():
#     output = request.form.to_dict()
#     print(output)
#     name = output["name"]
#     return render_template('index.html', predection=name)
### RESPONSE FROM LIVE FOR HTML DISPLAY
# @app.route('/receive_classpass', methods=['POST'])
# def receive_classpass():
#     data = request.get_json()
#     classpass = data['classpass']
#     print(f"Prediction for {classpass}")
#     return render_template('temp.html', prediction=classpass)
#     if data:
#         return jsonify({'success': True}), 200
#     else:
#         return jsonify({'error': 'No data received.'}), 400



def preprocess_image(image):
    input_size = (256, 256)
    resized_image = cv2.resize(image, input_size)
    normalized_image = resized_image / 255.0
    input_image = normalized_image.reshape((1,) + normalized_image.shape)
    return input_image
def predict_real_time(model, threshold=0.82, padding_factor=0.2):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_Cap.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
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

            if is_fake:
                prediction_value = round(prediction[0][0], 2)
                display_text = f"Fake ({prediction_value})"
                cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                prediction_value = round(prediction[0][0], 2)
                display_text = f"Real ({prediction_value})"
                cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()
@app.route('/video_feed')
def video_feed():
    meso = Meso4()
    meso.load('Meso4_DF')
    return Response(predict_real_time(meso),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_waveform_image(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Generate the waveform image
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, len(y) / sr, len(y)), y)
    ax.axis('off')  # Turn off axis
    plt.close(fig)  # Close the plot to avoid rendering it

    # Convert the plot to an image
    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)

    return output




if __name__ == "__main__":
    app.run(debug=False)