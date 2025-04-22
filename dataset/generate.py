import wave
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import tqdm
import random
import os
from datasets import load_dataset
import sounddevice as sd
import argparse
import numpy as np
import torchaudio
import torch
import librosa


VOICES = ["af_sky", "af_heart",]

# ✅ Positive phrases (keyword variants)
POSITIVE_SENTENCES = [
    "bonjour Lilio", "Lilio t'es là ?", "ok Lilio", "eh Lilio",
    "Dit Lilio",
]


def generateNegativeDataset():
    # === Paramètres ===
    OUTPUT_DIR = "./dataset/negative/auto"
    NB_SAMPLES = 500
    TARGET_DURATION = 2.0  # en secondes
    TARGET_SR = 16000
    TARGET_LEN = int(TARGET_DURATION * TARGET_SR)

    # === Créer le dossier de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === Charger le dataset Common Voice ou Fleurs français
    cv = load_dataset("google/fleurs", "fr_fr", split=f"train[:{NB_SAMPLES}]", trust_remote_code=True)


    # === Parcourir les échantillons
    for i, sample in enumerate(tqdm.tqdm(cv, desc="🔉 Découpe des fichiers audio")):
        if (i >= NB_SAMPLES):
            break
        audio = sample["audio"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
        sample_rate = audio["sampling_rate"]

        # Resample à 16kHz
        if sample_rate != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=TARGET_SR)

        total_len = waveform.shape[1]
        num_segments = total_len // TARGET_LEN

        for j in range(num_segments):
            start = j * TARGET_LEN
            end = start + TARGET_LEN
            segment = waveform[:, start:end]

            output_path = os.path.join(OUTPUT_DIR, f"negative_{i:06d}.wav")
            torchaudio.save(output_path, segment, sample_rate=TARGET_SR)

def auto_generate_audio(pipeline : KPipeline,phrases, directory, count, prefix):
    len_custom = len(os.listdir("dataset/positive/normal"))
    for i in tqdm.tqdm(range(count), desc=f"Generating {prefix} samples"):
        text = random.choice(phrases)
        voice = random.choice(VOICES)
        generator = pipeline(text, voice=voice)
        filename = f"{prefix}_{len_custom + i:04d}.wav"
        filepath = os.path.join(directory, filename)
        for i, (gs, ps, audio) in enumerate(generator):
            print(i, gs, ps)
            duration = len(audio) / 24000
            if duration > 2.0:
               audio = audio[:int(2.0 * 24000)]
               print("Truncated to 2 seconds")
               print(audio.shape)
               print(audio)
            else:
                print("Need to pad")
                audio = np.pad(audio, (0, int(2.0 * 24000) - len(audio)), mode='constant')
            
            display(Audio(data=audio, rate=24000, autoplay=i==0))
            sf.write(filepath, audio, 24000)

def generate_custom_audio(path="dataset/positive/normal"):
    FS = 16000  # fréquence d'échantillonnage
    DURATION = 2  # secondes
    len_custom = len(os.listdir(path))
    for i, phrase in enumerate(POSITIVE_SENTENCES):
        input(f"\n🎤 Dis maintenant : \"{phrase}\" (appuie sur Entrée quand prêt)")
        recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
        sd.wait()

        filename = f"{path}/positive_{len_custom + i:04d}.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(FS)
            wf.writeframes(recording.tobytes())

        print(f"✔️ Enregistré : {filename}")
    

def init():
    output_dir = 'dataset'
    positive_dir = os.path.join(output_dir, 'positive')
    negative_dir = os.path.join(output_dir, 'negative')
    os.makedirs(positive_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)


def main():
    # Define output directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", default=False, type=bool)
    parser.add_argument("--type", default="positive", type=str, choices=["positive", "negative"],)
    parser.add_argument("--test", default=False, type=bool)
    
    args = parser.parse_args()
    init()
    if args.auto:
        pipeline = KPipeline(lang_code='fr-fr', repo_id='hexgrad/Kokoro-82M')
        if args.type == "positive":
            auto_generate_audio(pipeline,POSITIVE_SENTENCES, "dataset/positive/auto", 100, args.type)
        elif args.type == "negative":
            generateNegativeDataset()
        
    else:
        if args.type == "positive":
            generate_custom_audio()
        elif args.type == "negative":
            print("Please use auto mode for negative samples")

main()