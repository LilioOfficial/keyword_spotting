# torch_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import os


class TorchModel(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        c, h, w = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(32 * ((h - 1) // 2) * ((w - 1) // 2), 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1)  # For BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x)


def load_model_from_checkpoint(path: str, device=None) -> nn.Sequential:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # You must know your input size and class count beforehand or save them in metadata
    input_channels = 3
    input_height = 50
    input_width = 50
    shape = (input_channels,input_height,input_width)

    model = TorchModel(shape).to(device)

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model