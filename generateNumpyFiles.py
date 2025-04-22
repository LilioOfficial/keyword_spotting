# build_npy_dataset.py
import os
import librosa
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing as mp
from preprocessing import extract_features  # import from your module

AUDIO_DIR = "dataset"
OUTPUT_DIR = "numpyFiles"
MAX_FILES = 1000
SR = 24000

def process_wav_to_tensor(wav_path):
    try:
        y, _ = librosa.load(wav_path, sr=SR)
        tensor = extract_features(y).squeeze(0).numpy()  # [3, 32, 32]
        return tensor
    except Exception as e:
        print(f"[ERROR] {wav_path} → {e}")
        return None

def process_folder(label, label_test):
    input_folder = os.path.join(AUDIO_DIR, label)
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".wav")])[:MAX_FILES*2]
    train_files = files[:MAX_FILES]
    test_files = files[MAX_FILES:]

    print(f"🔧 {label}: {len(train_files)} train, {len(test_files)} test")

    def make_tasks(file_list, subfolder):
        return [os.path.join(input_folder, f) for f in file_list]

    # Parallel processing
    with mp.Pool(mp.cpu_count()) as pool:
        train_data = list(tqdm(pool.imap(process_wav_to_tensor, make_tasks(train_files, label)), total=len(train_files), desc=f"{label} Train"))
        test_data = list(tqdm(pool.imap(process_wav_to_tensor, make_tasks(test_files, label_test)), total=len(test_files), desc=f"{label} Test"))

    train_data = np.array([x for x in train_data if x is not None], dtype=np.float32)
    test_data = np.array([x for x in test_data if x is not None], dtype=np.float32)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, f"{label}.npy"), train_data)
    np.save(os.path.join(OUTPUT_DIR, f"{label_test}.npy"), test_data)
    print(f"✅ Saved {label}.npy {train_data.shape} and {label_test}.npy {test_data.shape}")

if __name__ == "__main__":
    process_folder("positive", "positiveTest")
    process_folder("negative", "negativeTest")
