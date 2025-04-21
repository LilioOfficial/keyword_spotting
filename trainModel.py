# torch_model.py
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torchtnt.utils.loggers import CSVLogger
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch_model import TorchModel
# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data_from_npy(path):
    pos = np.load(os.path.join(path, "positive.npy"))
    neg = np.load(os.path.join(path, "negative.npy"))

    X = np.concatenate([pos, neg], axis=0)
    Y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))], axis=0)

    return X, Y


def main():
    pathData = './numpyFiles'
    trainRatio = 0.8
    epochs = 1000
    batch_size = 16
    earlyStopPatience = 5
    log_path = './logs/torch_model.csv'
    checkpoint_path = './trainedModel/best_model.pt'

    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./trainedModel', exist_ok=True)

    csv_logger = CSVLogger(log_path)

    # Load from .npy
    X, Y = load_data_from_npy(pathData)
    dimension = X.shape[1:]  # H, W, C

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=trainRatio)

    x_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
    x_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train = torch.tensor(Y_train, dtype=torch.float32)
    y_test = torch.tensor(Y_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TorchModel((dimension[2], dimension[0], dimension[1])).to(device)
    optimizer = optim.Adamax(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience_counter = earlyStopPatience

    history_loss = []
    history_acc = []

    for epoch in trange(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(test_loader)
        accuracy = correct / total

        csv_logger.log("loss", val_loss, step=epoch + 1)
        csv_logger.log("accuracy", accuracy, step=epoch + 1)

        history_loss.append(val_loss)
        history_acc.append(accuracy)

        print(f"Epoch {epoch + 1}: Val Loss = {val_loss:.4f}, Accuracy = {accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = earlyStopPatience
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter -= 1
            print(f"Patience left: {patience_counter}")
            if patience_counter == 0:
                print("Early stopping triggered")
                break

    # Plot loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_loss, label="Val Loss")
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_acc, label="Accuracy")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./logs/metrics_plot.png")
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "positive"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("./logs/confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()
