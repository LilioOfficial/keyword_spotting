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
MAX_FILES = 300
SR = 24000

def process_wav_to_tensor(wav_path):
    try:
        y, _ = librosa.load(wav_path, sr=SR)
        tensor = extract_features(y).squeeze(0).numpy()  # [3, 32, 32]
        return tensor
    except Exception as e:
        print(f"[ERROR] {wav_path} → {e}")
        return None

def process_folder(files, label_test, label):
    train_files = files[:MAX_FILES]
    test_files = files[MAX_FILES:]

    print(f"🔧 {label}: {len(train_files)} train, {len(test_files)} test")

    # Parallel processing
    with mp.Pool(mp.cpu_count()) as pool:
        train_data = list(tqdm(pool.imap(process_wav_to_tensor, train_files), total=len(train_files), desc=f"{label} Train"))
        test_data = list(tqdm(pool.imap(process_wav_to_tensor, test_files), total=len(test_files), desc=f"{label} Test"))

    train_data = np.array([x for x in train_data if x is not None], dtype=np.float32)
    test_data = np.array([x for x in test_data if x is not None], dtype=np.float32)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, f"{label}.npy"), train_data)
    np.save(os.path.join(OUTPUT_DIR, f"{label_test}.npy"), test_data)
    print(f"✅ Saved {label}.npy {train_data.shape} and {label_test}.npy {test_data.shape}")

if __name__ == "__main__":
    #join all files from subdirs
    list_pos_files = []
    dir = os.listdir(f'{AUDIO_DIR}/positive')
    for i in dir:
        file_names = os.listdir(f'{AUDIO_DIR}/positive/{i}')
        for j in file_names:
            list_pos_files.append(f'{AUDIO_DIR}/positive/{i}/{j}')
        
    list_neg_files = []
    dir = os.listdir(f'{AUDIO_DIR}/negative')
    for i in dir:
        if (i.endswith(".wav")):
            list_neg_files.append(f'{AUDIO_DIR}/negative/{i}')
        else:
            print(f"🔧 {i} is not a .wav file")
            file_names = os.listdir(f'{AUDIO_DIR}/negative/{i}')
            for j in file_names:
                list_neg_files.append(f'{AUDIO_DIR}/negative/{i}/{j}')

    print(f"🔧 Positive: {len(list_pos_files)} files"
          f"\n🔧 Negative: {len(list_neg_files)} files")

    process_folder(list_pos_files,label_test= "positiveTest", label="positive")
    process_folder( list_neg_files, label="negative", label_test="negativeTest")
