# torch_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import torch.nn.functional as F

# ---- Hyperparamètre global pour la résolution du spectrogramme ----
N_MEL = 32  # Change cette valeur selon tes tests (32, 64, etc.)
# device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
# Paths
checkpoint_path = './trainedModel/best_model.pt'

# 📁 waveform_cnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# 👇 Remplace le contenu de WaveformCNN dans waveform_cnn_model.py
class WaveformCNN(nn.Module):
    def __init__(self, input_length=48000):
        super(WaveformCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=80, stride=4, padding=38),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Binary output (logit)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ✅ Exemple d'usage
if __name__ == "__main__":
    model = WaveformCNN(input_length=48000)
    dummy = torch.randn(8, 1, 48000)  # batch de 8 waveforms 2 sec @ 24kHz
    out = model(dummy)
    print("Output shape:", out.shape)  # (8, 1)


class KeywordCNN(nn.Module):
    def __init__(self, input_channels=1, input_height=32, input_width=32):
        super(KeywordCNN, self).__init__()
    
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Calcul dynamique de la taille du flatten
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            print(f"Input shape: {dummy_input.shape}")
            x = self.pool1(F.relu(self.conv1(dummy_input)))
            print(f"Shape after conv1: {x.shape}")
            x = self.pool2(F.relu(self.conv2(x)))
            print(f"Shape after conv2: {x.shape}")
            x = self.pool3(F.relu(self.conv3(x)))
            print(f"Shape after conv3: {x.shape}")
            self.flat_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flat_size, 32)
        self.fc2 = nn.Linear(32, 1)  # Sortie binaire (logit)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logit



def load_model_from_checkpoint() -> nn.Sequential:
   # Load model
   # Params
    input_channels = 3   # or 1 depending on how you trained
    input_height = 32
    input_width = 32
    model = KeywordCNN(input_channels=input_channels, input_height=input_height, input_width=input_width)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()

    return model