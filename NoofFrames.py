import cv2

def count_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video capture object
    cap.release()

    return total_frames

# Example usage
video_path = "ip.mp4"
num_frames = count_frames(video_path)

if num_frames is not None:
    print(f"Number of frames in the video: {num_frames}")
else:
    print("Error counting frames.")
