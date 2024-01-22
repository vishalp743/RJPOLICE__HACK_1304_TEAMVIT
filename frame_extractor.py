import cv2
import os


def extract_frames(video_path, output_folder, frame_interval=50):
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
    # Specify the path to the video file
    video_path = "IP1.mp4"

    # Specify the output folder for frames
    output_folder = "Frames"

    # Specify the frame interval
    frame_interval = 50

    # Call the function to extract frames
    extract_frames(video_path, output_folder, frame_interval)
