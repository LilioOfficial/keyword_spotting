# torch_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import os


class TorchModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        channels, height, width = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(32 * ((height - 1) // 2) * ((width - 1) // 2), 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
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
    num_classes = 2
    shape = (input_channels,input_height,input_width)

    model = TorchModel(shape, num_classes).to(device)

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model