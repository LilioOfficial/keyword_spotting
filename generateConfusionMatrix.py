# IMPORT
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import librosa
import librosa.display
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch_model import load_model_from_checkpoint  # assumes you've created this function in your torch model file

def generateMatrix(model, datasetTestPath, imageSize, destinationMatrix, device):
    y_true = []
    y_pred = []

    transform = transforms.Compose([
        transforms.Resize(imageSize),
        transforms.ToTensor()
    ])

    model.eval()

    print('\nEvaluation :')
    for root, dirs, files in os.walk(datasetTestPath):
        for label_index, mydir in enumerate(dirs):
            for sample in tqdm(os.listdir(os.path.join(root, mydir)), f"Prédiction de la classe '{mydir}'"):
                sample_path = os.path.join(root, mydir, sample)
                y, sr = librosa.load(sample_path)
                spec = librosa.feature.melspectrogram(y=y, sr=sr)
                librosa.display.specshow(librosa.power_to_db(spec, ref=np.max))
                canvas = plt.get_current_fig_manager().canvas
                canvas.draw()
                img = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_argb())
                plt.close()

                img = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img).squeeze(1)  # shape [1]
                    prob = torch.sigmoid(output).item()
                    pred = 1 if prob > 0.5 else 0

                y_true.append(label_index)
                y_pred.append(pred)

    cnf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cnf_matrix) / np.sum(cnf_matrix) * 100.
    print('\nPrécision : {0:.3f}%'.format(accuracy))

    np.set_printoptions(precision=2)
    plt.figure()

    cmap = plt.cm.Blues
    classes = sorted(os.listdir(datasetTestPath))
    title = 'Confusion matrix'

    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    os.makedirs(destinationMatrix, exist_ok=True)
    plt.savefig(os.path.join(destinationMatrix, 'confusionMatrix.png'))

def main():
    modelPath = './trainedModel/best_model.pt'
    datasetTestPath = './datasetTest'
    destinationMatrix = './graph'
    imageSize = (50, 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_checkpoint(modelPath, device)
    generateMatrix(model, datasetTestPath, imageSize, destinationMatrix, device)

if __name__ == "__main__":
    main()
