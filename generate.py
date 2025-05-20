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
from augment import augment_folder


NAME = ["ludo", "leo", "bene", "benji", "christian", "alix", "tom", "auto"]

VOICES = ["af_sky", "af_heart", "af_jessica", "af_nicole"]

# ✅ Positive phrases (keyword variants)
SENTENCES = {
    "positive" : [
    "Lilio", "Lilioh", "Liliooo", "Lilo", "Liliot", "Liliau", 
    "Liliose", "Lilleau", "Lillio", "Lilio", "Liliooo",
    "Lili haut", "Lili oh", "Lili yo", "Lili yooh", "Lilyo"
],
    
    #     "positive" : [ "bonjour Lilio", "Lilio t'es là ?", "ok Lilio", "eh Lilio",
    # "Dit Lilio",]
    # ,
    
        "negative" : [
    # Variantes phonétiques proches
    "Lili", "Lilo", "Liloé", "Lia", "Liya", "Lina", "Linao", "Lino", "Liloah",
    "Léo", "Liyao", "Lilou", "Lia oh", "Lili haut", "Lili oh", "Lilioh",
    
    # Confusions syllabiques
    "Lali", "Lila", "Lalia", "Lalya", "Lelo", "Leloa", "Liloï", "Lya", "Lio",
    
    # Autres noms ou sons ressemblants
    "Linoé", "Léa", "Léon", "Rio", "Lyo", "Nino", "Lioh", "Yoyo", "Liyao",
    
    # Phrases pièges
    "Je crois que c'était Lilou",  
    "Dis Lili haut",  
    "Tu veux dire Lilo ?",  
    "C’est qui Lili ?",  
    "Lino est rentré",  
    "Léo vient de passer",  
    "Lila a appelé tout à l’heure"
        ]
    }


def generateNegativeDataset():
    # === Paramètres ===
    OUTPUT_DIR = "./dataset/negative/google"
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
    len_custom = len(os.listdir(directory))
    for i in tqdm.tqdm(range(count), desc=f"Generating {prefix} samples"):
        text = random.choice(phrases)
        voice = random.choice(VOICES)
        generator = pipeline(text, voice=voice)
        filename = f"{voice}_{len_custom + i:04d}.wav"
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

def generate_custom_audio(path, sentences):
    FS = 24000  # fréquence d'échantillonnage
    DURATION = 2  # secondes
    len_custom = len(os.listdir(path))
    for i, phrase in enumerate(sentences):
        input(f"\n🎤 Dis maintenant : \"{phrase}\" (appuie sur Entrée quand prêt)")
        recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
        sd.wait()

        filename = f"{path}positive_{len_custom + i:04d}.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(FS)
            wf.writeframes(recording.tobytes())

        print(f"✔️ Enregistré : {filename}")
    



def main():
    # Define output directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, choices=NAME, help="Name of the person")
    parser.add_argument("--test", type=bool, default= False, help="weither it's test or not")
    args = parser.parse_args()

    match args.test:
        case True :
            dir_pos = f'dataset/test/positive/{args.name}/normal/'
            dir_aug = f'dataset/test/positive/{args.name}/aug/'
            dir_neg = f'dataset/test/negative/{args.name}/normal/'
            dir_aug_neg = f'dataset/test/negative/{args.name}/aug/'
            os.makedirs(dir_neg, exist_ok=True)
            os.makedirs(dir_pos, exist_ok=True)
        case False :
            dir_pos = f'dataset/positive/{args.name}/normal/'
            dir_aug = f'dataset/positive/{args.name}/aug/'
            dir_neg = f'dataset/negative/{args.name}/normal/'
            dir_aug_neg = f'dataset/negative/{args.name}/aug/'
            os.makedirs(dir_neg, exist_ok=True)
            os.makedirs(dir_pos, exist_ok=True)
    match args.name :
        case "auto" :
            pipeline = KPipeline(lang_code='fr-fr', repo_id='hexgrad/Kokoro-82M')
            auto_generate_audio(pipeline,SENTENCES["positive"],dir_pos, 500, "positive")
            auto_generate_audio(pipeline,SENTENCES["negative"],dir_neg, 1500, "negative")
        case _:
            generate_custom_audio(path=dir_pos, sentences=SENTENCES["positive"])
            generate_custom_audio(path=dir_neg, sentences=SENTENCES["negative"])
    
    print("Starting augmentation")
    # Generate augmented files
    augment_folder(dir_pos, dir_aug)
    augment_folder(dir_neg, dir_aug_neg)


if __name__ == '__main__':
    main()