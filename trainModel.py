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
from torch_model import TorchModel

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_labels(path):
    labels = [file.replace('.npy', '') for file in os.listdir(path) if file.endswith('.npy')]
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, torch.nn.functional.one_hot(torch.tensor(label_indices)).numpy()


def get_train_test(train_ratio, pathData):
    labels, _, _ = get_labels(pathData)
    classNumber = 0

    X = data = np.load(pathData + '/' + labels[0] + '.npy')
    Y = np.zeros(X.shape[0])
    dimension = X[0].shape
    classNumber += 1

    for i, label in enumerate(labels[1:]):
        data = np.load(pathData + '/' + label + '.npy')
        X = np.vstack((X, data))
        Y = np.append(Y, np.full(data.shape[0], fill_value=(i+1)))
        classNumber += 1
    print(classNumber)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_ratio)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor, classNumber, dimension



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

    x_train, x_test, y_train, y_test, classNumber, dimension = get_train_test(trainRatio, pathData)

    x_train = x_train.permute(0, 3, 1, 2)
    x_test = x_test.permute(0, 3, 1, 2)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TorchModel((dimension[2], dimension[0], dimension[1]), classNumber).to(device)
    optimizer = optim.Adamax(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = earlyStopPatience

    for epoch in trange(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(test_loader)
        accuracy = correct / total

        csv_logger.log("loss", val_loss, step=epoch+1)
        csv_logger.log("accuracy", accuracy, step=epoch+1)

        print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}, Accuracy = {accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = earlyStopPatience
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print("Early stopping triggered")
                break

if __name__ == "__main__":
    main()