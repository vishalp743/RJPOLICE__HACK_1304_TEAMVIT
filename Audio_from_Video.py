from moviepy.video.io.VideoFileClip import VideoFileClip
sr=16000
def video_to_audio(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)


from moviepy.editor import VideoFileClip
import librosa
import numpy as np
from scipy.io import wavfile
from noisereduce import reduce_noise
from pydub import AudioSegment

def mp3_to_wav(input_path, output_path):
    audio = AudioSegment.from_mp3(input_path)
    audio.export(output_path, format="wav")

def preprocess_audio(input_path, output_path, sr=16000):

    # Check if the input is an MP3 file and convert it to WAV
    if input_path.lower().endswith('.mp3'):
        mp3_to_wav(input_path, "temp.wav")
        input_path = "temp.wav"
    # Load audio file
    audio, sr = librosa.load(input_path, sr=sr)

    # Perform noise reduction using noisereduce library
    reduced_audio = reduce_noise(audio, sr=sr)

    # Normalize audio
    normalized_audio = librosa.util.normalize(reduced_audio)

    # Save the preprocessed audio
    wavfile.write(output_path, sr, (normalized_audio * 32767).astype(np.int16))

def video_to_audio(video_path, audio_output_path):
    # Extract audio from video and save as WAV
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_output_path, codec='pcm_s16le', ffmpeg_params=["-ac", "1"])

    # Preprocess the extracted audio
    preprocess_audio(audio_output_path, audio_output_path, sr=sr)

