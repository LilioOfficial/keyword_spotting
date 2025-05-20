# 📁 evaluate_waveform.py
import argparse
import os
import librosa
import sounddevice as sd
import torch
import numpy as np
import time
from models import WaveformCNN

# ---- CONFIG ----
MODEL_PATH = "trainedModel/waveform_model.pt"
SAMPLE_RATE = 24000
DURATION = 2.0  # seconds
INPUT_LENGTH = int(SAMPLE_RATE * DURATION)
THRESHOLD = 0.5

# ---- Load model ----
def load_model():
    model = WaveformCNN(input_length=INPUT_LENGTH)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# ---- Preprocessing ----
def preprocess_audio(audio):
    if len(audio) < INPUT_LENGTH:
        pad = INPUT_LENGTH - len(audio)
        audio = np.pad(audio, (0, pad), mode='constant')
    else:
        audio = audio[:INPUT_LENGTH]
        
    tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, len]
    return tensor

# ---- Evaluate a single waveform ----
def evaluate_tensor(tensor, model):
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
        print(f"🔍 Probability: {prob:.4f}")
        if prob > THRESHOLD:
            print("✅ Keyword detected!")
            return True
        else:
            print("❌ No keyword detected.")
            return False

# ---- Evaluate from .wav file ----
def from_wav_file(path, model):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    audio = audio.flatten()
    tensor = preprocess_audio(audio)
    evaluate_tensor(tensor, model)

# ---- Continuous evaluation from mic ----
def continuous_mic(model):
    print("🎤 Speak naturally. Keyword detection is live. Press Ctrl+C to stop.")
    try:
        while True:
            audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            audio = audio.flatten()
            tensor = preprocess_audio(audio)
            evaluate_tensor(tensor, model)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("🛑 Stopped.")

# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, help="Path to a .wav file")
    parser.add_argument("--live", action="store_true", help="Run continuous microphone detection")
    args = parser.parse_args()

    model = load_model()

    if args.wav:
        from_wav_file(args.wav, model)
    elif args.live:
        continuous_mic(model)
    else:
        print("❗ Use --wav <file.wav> or --live")
