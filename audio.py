import torch
import torch.nn.functional as F
import torchaudio
import librosa
import os
from scipy.io.wavfile import read
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead


def load_audio(audiopath, sampling_rate=22000):
    if audiopath.endswith('.wav'):
        audio, sr = librosa.load(audiopath, sr=sampling_rate)
        audio = torch.FloatTensor(audio)
    else:
        assert False, f"unsupported audio format provided: {audiopath[-4:]}"

    # Remove any channel data
    if sr != sampling_rate:
        audio = torchaudio.functional.resample(audio, sr, sampling_rate)

    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with audio data. Max={audio.max()} min={audio.min()}")
        audio.clip_(-1, 1)

    return audio.unsqueeze(0)


def classify_audio_clip(clip):
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                    resnet_blocks=2, attn_blocks=4, num_attn_heads=4,
                                                    base_channels=32, dropout=0, kernel_size=5,
                                                    distribute_zero_label=False)

    # Load the pre-trained classifier model
    state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]


def main():
    os.listdir()
    audio_file_path = "output_audio.wav"
    audio_clip = load_audio(audio_file_path)

    result = classify_audio_clip(audio_clip)
    result = result.item()

    print(f"The uploaded audio is {result * 100:.2f}% likely to be AI-generated")


if __name__ == "__main__":
    main()
