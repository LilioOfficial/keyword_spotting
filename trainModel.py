# torch_model.py
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchtnt.utils.loggers import CSVLogger
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch_model import KeywordCNN

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load .npy data
def load_data_from_npy(path):
    pos = np.load(os.path.join(path, "positive.npy"))
    neg = np.load(os.path.join(path, "negative.npy"))
    print(f"Loaded {len(pos)} positive and {len(neg)} negative samples")
    print(f"Positive shape: {pos.shape}, Negative shape: {neg.shape}")
    X = np.concatenate([pos, neg], axis=0)
    Y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))], axis=0)
    return X, Y

# Main training script
def main():
    pathData = './numpyFiles'
    trainRatio = 0.8
    epochs = 200
    batch_size = 16
    earlyStopPatience = 10
    log_path = './logs/torch_model.csv'
    checkpoint_path = './trainedModel/best_model.pt'

    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./trainedModel', exist_ok=True)

    csv_logger = CSVLogger(log_path)

    X, Y = load_data_from_npy(pathData)
    dimension = X.shape[1:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=trainRatio, stratify=Y)
    x_train = torch.tensor(X_train, dtype=torch.float32)
    x_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)

    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
    print(f"Train labels shape: {y_train.shape}, Test labels shape: {y_test.shape}")
    #Debug log 

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    model = KeywordCNN(input_channels=dimension[0], input_height=dimension[1], input_width= dimension[2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    pos_weight = torch.tensor([len(Y_train[Y_train==0]) / len(Y_train[Y_train==1])]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float('inf')
    patience_counter = earlyStopPatience
    history_loss, history_acc = [], []


    for epoch in trange(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # input: [batch_size, 32, 32, 3]
            # labels: [batch_size, 1]
            # print("inputs shape:", inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_loss /= len(test_loader)
        accuracy = correct / total
        history_loss.append(val_loss)
        history_acc.append(accuracy)

        csv_logger.log("val_loss", val_loss, step=epoch + 1)
        csv_logger.log("val_accuracy", accuracy, step=epoch + 1)
        print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}, Accuracy = {accuracy:.4f}")

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

    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_loss, label="Validation Loss")
    plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title("Val Loss"), plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(history_acc, label="Accuracy")
    plt.xlabel("Epoch"), plt.ylabel("Accuracy"), plt.title("Val Accuracy"), plt.grid()
    plt.tight_layout()
    plt.savefig("./logs/metrics_plot.png")
    plt.show()

    cm = confusion_matrix(np.array(all_targets).flatten(), np.array(all_preds).flatten())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "positive"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("./logs/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    main()
