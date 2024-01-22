import cv2

# Load the face cascade classifier
face_Cap = cv2.CascadeClassifier(r"C:\Users\Vishal\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

# Open the video capture
video_cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video
    ret, video_data = video_cap.read()

    # Convert the frame to grayscale
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_Cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Create a mask for the face region
    mask = col.copy()
    mask[:] = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), thickness=cv2.FILLED)

    # Apply a blur effect to the background
    blurred_background = cv2.GaussianBlur(video_data, (15, 15), 0)
    video_data = cv2.bitwise_and(blurred_background, blurred_background, mask=cv2.bitwise_not(mask))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Live Video", video_data)

    # Break the loop if the key 'a' is pressed
    if cv2.waitKey(10) == ord("a"):
        break

# Release the video capture
video_cap.release()
cv2.destroyAllWindows()
