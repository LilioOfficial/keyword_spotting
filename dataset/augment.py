# 📁 audio_augmentation.py
import librosa
import numpy as np
import random
import os
import soundfile as sf
import multiprocessing as mp
from tqdm import tqdm

# ---- Configuration par défaut ----
SR = 24000
AUG_PER_FILE = 3

# ---- Augmentation primitives ----
def add_background_noise(audio, noise_level=0.005):
    noise = np.random.normal(0, 1, len(audio))
    return audio + noise_level * noise

def shift_audio(audio, max_shift=0.2):
    shift = int(random.uniform(-max_shift, max_shift) * len(audio))
    return np.roll(audio, shift)

def stretch_audio(audio, rate_range=(0.8, 1.2)):
    rate = random.uniform(*rate_range)
    return librosa.effects.time_stretch(audio, rate)

def pitch_shift_audio(audio, sr=SR, pitch_range=(-2, 2)):
    n_steps = random.uniform(*pitch_range)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def drop_chunks(audio, drop_fraction=0.1):
    length = len(audio)
    chunk_size = int(length * drop_fraction)
    start = random.randint(0, length - chunk_size)
    audio[start:start + chunk_size] = 0
    return audio

def apply_random_augmentation(audio, sr=SR):
    funcs = [add_background_noise, shift_audio, stretch_audio, pitch_shift_audio, drop_chunks]
    n = random.randint(1, 3)
    selected = random.sample(funcs, n)
    for func in selected:
        try:
            audio = func(audio) if func != pitch_shift_audio else func(audio, sr)
        except Exception as e:
            print(f"[⚠️  Warning] {func.__name__} failed: {e}")
    return audio

# ---- Process single file ----
def augment_file(args):
    in_path, out_folder, n_aug = args
    try:
        y, _ = librosa.load(in_path, sr=SR)
        base = os.path.splitext(os.path.basename(in_path))[0]
        for i in range(n_aug):
            aug = apply_random_augmentation(np.copy(y))
            out_path = os.path.join(out_folder, f"{base}_aug{i+1}.wav")
            sf.write(out_path, aug, SR)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to process {in_path}: {e}")
        return False

# ---- Batch processing ----
def augment_folder(input_dir, output_dir, n_aug_per_file=AUG_PER_FILE):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    tasks = [(os.path.join(input_dir, f), output_dir, n_aug_per_file) for f in files]

    with mp.Pool(mp.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(augment_file, tasks), total=len(tasks), desc="Augmenting"))

    print(f"✅ Augmented {len(tasks)} files x{n_aug_per_file} saved to {output_dir}")

# 🔁 Exemple d'utilisation
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input folder with .wav files")
    parser.add_argument("--output", type=str, required=True, help="Output folder for augmented .wav files")
    parser.add_argument("--n", type=int, default=3, help="Number of augmentations per file")
    args = parser.parse_args()

    augment_folder(args.input, args.output, n_aug_per_file=args.n)