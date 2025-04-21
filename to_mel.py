import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm
import numpy as np

AUDIO_DIR = "dataset"
OUTPUT_DIR = "spectrograms"
IMG_SIZE = (224, 224)  # Or (50, 50) depending on your model

def create_spectrogram(wav_path, save_path):
    y, sr = librosa.load(wav_path, sr=24000)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(IMG_SIZE[0]/100, IMG_SIZE[1]/100), dpi=100)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_folder(label):
    input_folder = os.path.join(AUDIO_DIR, label)
    output_folder = os.path.join(OUTPUT_DIR, label)
    os.makedirs(output_folder, exist_ok=True)

    for fname in tqdm(os.listdir(input_folder), desc=f"Processing {label}"):
        if fname.endswith(".wav"):
            in_path = os.path.join(input_folder, fname)
            out_path = os.path.join(output_folder, fname.replace(".wav", ".png"))
            create_spectrogram(in_path, out_path)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_folder("positive")
    process_folder("negative")
