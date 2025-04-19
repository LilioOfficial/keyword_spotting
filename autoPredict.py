# voice_predictor_web.py
import os
import time
import wave
import torch
import librosa
import librosa.display
import numpy as np
from PIL import Image
from tqdm import tqdm
from aiohttp import web
import matplotlib.pyplot as plt
from torchvision import transforms
from torch_model import load_model_from_checkpoint

# === Audio parameters ===
FORMAT = 8  # corresponds to pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2
SOUND_PATH = './testAudio/test.wav'
SOUND_SIZE = (50, 50)
MODEL_PATH = './trainedModel/best_model.pt'
LABELS = ['chat', 'chien']

# === Model and device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_from_checkpoint(MODEL_PATH, device)

# === Image preprocessing ===
transform = transforms.Compose([
    transforms.Resize(SOUND_SIZE),
    transforms.ToTensor()
])


def predict_from_audio(path_to_wav: str):
    y, sr = librosa.load(path_to_wav)
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(spec, ref=np.max))

    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    plt.tight_layout(pad=0)
    plt.axis('off')
    img = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_argb())
    plt.close()

    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        prediction = torch.softmax(output, dim=1)[0].cpu().numpy()

    max_index = np.argmax(prediction)
    predicted_label = LABELS[max_index]
    confidence = prediction[max_index] * 100

    return predicted_label, confidence, prediction


async def handle_predict(request):
    data = await request.post()
    audio_file = data['audio'].file

    with open(SOUND_PATH, 'wb') as f:
        f.write(audio_file.read())

    label, conf, all_preds = predict_from_audio(SOUND_PATH)
    print(all_preds)

    return web.json_response({
        "label": label,
        "confidence": f"{conf:.2f}%",
        "all_predictions": {LABELS[i]: float(f"{p * 100:.2f}") for i, p in enumerate(all_preds)}
    })


async def index(request):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Classifier</title>
    </head>
    <body>
        <h1>Upload a WAV file for classification</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="audio" accept="audio/wav" required>
            <br><br>
            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    """
    return web.Response(text=html, content_type='text/html')


app = web.Application()
app.router.add_get('/', index)
app.router.add_post('/predict', handle_predict)

if __name__ == '__main__':
    os.makedirs('./testAudio', exist_ok=True)
    web.run_app(app, port=8080)
