from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import tqdm
import random
import os

voices = ["af_sky", "af_heart",
"af_alloy",
"af_aoede",
"af_bella",
"af_jessica",
"af_kore",
"af_nicole",
"af_nova",
"af_river",
"af_sarah",
"af_sky",
"am_adam",
"am_echo",
"am_eric",
"am_fenrir",
"am_liam",
"am_michael",
"am_onyx",
"am_puck",
"am_santa"]



# ✅ Positive phrases (keyword variants)
positive_phrases = [
    "bonjour Lilio", "salut Lilio", "coucou Lilio", "hé Lilio", "ok Lilio",
    "Lilio tu m'entends ?", "Lilio réveille-toi", "hey Lilio", "allo Lilio",
    "Lilio tu es là ?", "hello Lilio", "hé Lilio ça va ?"
]

# ❌ Negative phrases (no keyword)
negative_phrases = [
    "bonjour Siri", "quelle heure est-il ?", "joue de la musique", "réveille-moi demain",
    "salut tout le monde", "peux-tu m’aider ?", "c’est l’heure de partir",
    "Léo est arrivé", "j’ai besoin d’un café", "ouvre la porte", "Lulu est parti"
]

def generate_audio(pipeline,phrases, directory, count, prefix):
    for i in tqdm.tqdm(range(count), desc=f"Generating {prefix} samples"):
        text = random.choice(phrases)
        voice = random.choice(voices)
        generator = pipeline(text, voice=voice)
        filename = f"{prefix}_{i:04d}.wav"
        filepath = os.path.join(directory, filename)
        for i, (gs, ps, audio) in enumerate(generator):
            print(i, gs, ps)
            display(Audio(data=audio, rate=24000, autoplay=i==0))
            sf.write(filepath, audio, 24000)


def main():
    # Define output directories
    output_dir = 'dataset'
    positive_dir = os.path.join(output_dir, 'positive')
    negative_dir = os.path.join(output_dir, 'negative')
    os.makedirs(positive_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)
    pipeline = KPipeline(lang_code='fr-fr', repo_id='hexgrad/Kokoro-82M')
    generate_audio(pipeline,positive_phrases, positive_dir, 1000, "positive")
    generate_audio(pipeline,negative_phrases, negative_dir, 1000, "negative")

main()