# 📁 train_waveform.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import WaveformCNN

# ---- CONFIG ----
POS_DIR = "./dataset/positive"
NEG_DIR = "./dataset/negative"
SAMPLE_RATE = 24000
DURATION = 2.0
INPUT_LENGTH = int(SAMPLE_RATE * DURATION)
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-5
PATIENCE = 5
os.makedirs("./graphs", exist_ok=True)

# ---- Device selection ----
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"🖥️ Using device: {DEVICE}")

# ---- Dataset ----
class RawAudioDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        audio, _ = librosa.load(path, sr=SAMPLE_RATE)
        if len(audio) < INPUT_LENGTH:
            pad = INPUT_LENGTH - len(audio)
            audio = np.pad(audio, (0, pad), mode='constant')
        else:
            audio = audio[:INPUT_LENGTH]
        # Normalize audio
        audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-6)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        return audio_tensor, torch.tensor([label], dtype=torch.float32)


# ---- Load dataset ----
def listFolders(path):
    list_pos_files = []
    dir = os.listdir(path)
    print(f"🔧 {path}: {len(dir)} files")
    for i in dir:
        if (i.endswith(".wav")):
            list_pos_files.append(f'{path}/{i}')
        else:
            list_pos_files.extend(listFolders(f'{path}/{i}'))
    return list_pos_files


def load_filepaths():
    pos_files = listFolders(POS_DIR)
    neg_files = listFolders(NEG_DIR)
    all_files = pos_files + neg_files
    labels = [1] * len(pos_files) + [0] * len(neg_files)
    print(f"🔧 Positive: {len(pos_files)} files\n🔧 Negative: {len(neg_files)} files")
    return train_test_split(all_files, labels, test_size=0.3,  stratify=labels, random_state=42, shuffle=True)

# ---- Training loop ----
def train():
    print("🚀 Starting training...")
    X_train, X_test, y_train, y_test = load_filepaths()
    train_dataset = RawAudioDataset(X_train, y_train)
    test_dataset = RawAudioDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = WaveformCNN(input_length=INPUT_LENGTH).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    history_loss = []
    history_acc = []
    best_acc = 0.0
    patience_counter = PATIENCE

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, targets = batch
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history_loss.append(avg_loss)
        print(f"📉 Epoch {epoch+1} Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                probas = torch.sigmoid(outputs).cpu().numpy()
                preds.extend((probas > 0.5).astype(int))
                targets.extend(labels.numpy().astype(int))

        acc = accuracy_score(targets, preds)
        history_acc.append(acc)
        print(f"✅ Test Accuracy: {acc:.4f}")

        # Early stopping
        if acc > best_acc:
            best_acc = acc
            patience_counter = PATIENCE
            torch.save(model.state_dict(), "waveform_model.pt")
            print("💾 Best model saved")
        else:
            patience_counter -= 1
            print(f"⏳ Patience left: {patience_counter}")
            if patience_counter == 0:
                print("🛑 Early stopping triggered")
                break

    # Save graphs
    plt.figure(figsize=(10, 4))
    plt.plot(history_loss, label='Loss')
    plt.plot(history_acc, label='Accuracy')
    plt.xlabel("Epoch")
    plt.title("Training Progress")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./graphs/training_curve.png")
    print("📈 Training graph saved to ./graphs/training_curve.png")

if __name__ == "__main__":
    train()
