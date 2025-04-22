import numpy as np
import librosa
import torch

def extract_features(audio, sr=24000, n_mels=32, n_frames=32):
    # 1. Log-mel
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel)

    # 2. Delta + Delta-Delta
    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)

    # 3. Truncate or pad to fixed width (time axis)
    def fix_width(m):
        if m.shape[1] < n_frames:
            return np.pad(m, ((0, 0), (0, n_frames - m.shape[1])), mode='constant')
        else:
            return m[:, :n_frames]

    logmel = fix_width(logmel)
    delta = fix_width(delta)
    delta2 = fix_width(delta2)

    # 4. Stack into 3 channels
    stacked = np.stack([logmel, delta, delta2], axis=0)  # [3, n_mels, n_frames]

    # 5. Normalize (optional, improves generalization)
    mean = np.mean(stacked)
    std = np.std(stacked) + 1e-6
    stacked = (stacked - mean) / std

    # 6. Convert to tensor [1, 3, H, W]
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
