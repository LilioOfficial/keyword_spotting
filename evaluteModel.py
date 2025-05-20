# ============================
# 📁 evaluate_file.py
# ============================
import argparse
import numpy as np
import sounddevice as sd
import torch
import librosa
import os
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from models import KeywordCNN
from preprocessing import extract_features
import soundfile as sf
import time

# ---- CONFIG ----
MODEL_PATH = "trainedModel/waveform_model.pt"
SR = 24000
DURATION = 2.0
THRESHOLD = 0.5

# ---- Load model ----
model = KeywordCNN(input_channels=3, input_height=32, input_width=32)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# ---- Inference from tensor ----
def evaluate_tensor(tensor):
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
    print(f"🔍 Probability: {prob:.4f}")
    if prob > THRESHOLD:
        print("✅ Keyword detected!")
    else:
        print("❌ No keyword detected.")

# ---- From microphone ----
def from_microphone():
    print("🎤 Speak now...")
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    print("🔊 Recording finished. Saving it")
    sf.write("testAudio/recording.wav", audio, SR)
    audio = audio.flatten()
    features = extract_features(audio)  # [1, 3, 32, 32]
    evaluate_tensor(features)

# ---- From raw .wav file ----
def from_wav_file(path):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return
    audio, _ = librosa.load(path, sr=SR)
    features = extract_features(audio)  # [1, 3, 32, 32]
    evaluate_tensor(features)

# ---- From test .npy files ----
def from_npy():
    X = np.load("./numpyFiles/positive.npy")
    Y = np.ones(len(X))
    X_neg = np.load("./numpyFiles/negative.npy")
    Y_neg = np.zeros(len(X_neg))
    X = np.concatenate([X, X_neg], axis=0)
    Y = np.concatenate([Y, Y_neg], axis=0)

    x_test = torch.tensor(X, dtype=torch.float32)
    y_test = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        outputs = model(x_test)
        probs = torch.sigmoid(outputs)
        preds = (probs > THRESHOLD).float()

    acc = accuracy_score(y_test.numpy(), preds.numpy())
    cm = confusion_matrix(y_test.numpy(), preds.numpy())

    print(f"✅ Accuracy: {acc:.4f}")
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"]).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()



# ---- Continuous microphone evaluation ----
def continuous_detection():
    print("🎤 Continuous mode — speak naturally. Say the keyword to trigger detection.")
    try:
        while True:
            audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
            sd.wait()
            audio = audio.flatten()
            features = extract_features(audio)
            detected = evaluate_tensor(features)
            if detected:
                print("🚨 Keyword detected!")
                sf.write("testAudio/recording.wav", audio, SR)
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("🛑 Stopped.")


# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic", action="store_true", help="Use microphone input")
    parser.add_argument("--npy", action="store_true", help="Evaluate full dataset")
    parser.add_argument("--wav", type=str, help="Path to a .wav file to evaluate")
    parser.add_argument("--cont", action="store_true", help="Continuous detection from microphone")
    args = parser.parse_args()

    if args.mic:
        from_microphone()
    elif args.npy:
        from_npy()
    elif args.wav:
        from_wav_file(args.wav)
    elif args.cont:
        continuous_detection()
    else:
        print("❗ Use one of: --mic | --npy | --wav <file> | --cont")
