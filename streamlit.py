import librosa
import matplotlib.pyplot as plt

def generate_and_save_waveform_plot():
    # Load the WAV file
    wav_file="output_audio.wav"
    output_image_path= "static/Images/output_image.png"
    librosa_audio_data, librosa_sample_rate = librosa.load(wav_file)

    # Plot the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(librosa_audio_data)
    plt.title('Waveform Plot')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Save the plot as an image
    plt.savefig(output_image_path)
    plt.close()  # Close the plot to avoid displaying it in the console

    print(f"Waveform plot saved successfully: {output_image_path}")
