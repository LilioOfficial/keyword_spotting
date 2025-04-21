# 📁 Dataset Generation for Keyword Spotting

This guide helps you create a **balanced dataset** for training a keyword spotter (e.g., detecting "Hello Lilio"). You’ll use **text-to-speech**, **mel-spectrogram conversion**, and optionally save as `.npy` for training a neural network.

---

## 🛠 Tools Required

- Python 3.8+
- `librosa`, `matplotlib`, `numpy`, `soundfile`
- (optional) a TTS model like **`hexgrad/Kokoro-82M`** or **Coqui TTS** for generating voices

---

## 📁 Dataset Structure

Organize your dataset as:

```
./dataset/
├── positive/        # Samples with "Hello Lilio"
└── negative/        # Samples with anything else
```

Each folder contains `.npy` files of mel-spectrograms.

---

## 🔊 Step 1: Generate Audio Samples

Use any TTS engine to generate `.wav` files:

```python
from transformers import pipeline
import soundfile as sf

# Load TTS pipeline (example)
tts = pipeline("text-to-speech", model="hexgrad/Kokoro-82M")

sentences = [
    "Bonjour Lilio", "Salut Lilio", "Hello Lilio", "Coucou Lilio",
    "Comment vas-tu Lilio", "Hey Lilio", "Lilio est là?"
]

for i, text in enumerate(sentences):
    output = tts(text, voice="af_sky")
    sf.write(f"positive/sample_{i}.wav", output["audio"], samplerate=16000)
```

Repeat with **non-keyword sentences** for the `negative` class.

---

## 🎛️ Step 2: Convert WAV to Mel-Spectrogram

```python
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "./dataset/positive"  # or "negative"
os.makedirs(output_dir, exist_ok=True)

for i, filename in enumerate(os.listdir("positive_wavs/")):
    y, sr = librosa.load(f"positive_wavs/{filename}")
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    np.save(f"{output_dir}/sample_{i}.npy", mel_db)
```

You now have `.npy` files ready for training.

---

## 🧠 Step 3: Train the Model

Use a CNN (e.g., `Conv2D + ReLU + MaxPool + Flatten + Dense + Sigmoid`) on mel-spectrograms loaded from `.npy` files.

---

## ✅ Tips

- Use **10-30 seconds** of varied examples per class for small models.
- Normalize all mel-spectrograms to the same shape (e.g., 50x50 or 64x64)
- Augment data with **noise**, **speed variation**, or **pitch shift**

---

## 📦 Optional Enhancements

- Save `train.csv` file mapping each `.npy` to its label
- Include speaker variation
- Use real recorded voices from `sounddevice` for robustness

---

## Questions?

Want to automate the whole pipeline? I can generate a script to batch-generate and convert everything for you. 😉

