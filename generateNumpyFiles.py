import os
import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_DIR = "spectrograms"
OUTPUT_DIR = "numpyFiles"
TARGET_SIZE = (50, 50)  # or (224, 224) depending on your model

def images_to_array(image_folder):
    image_list = []

    for fname in sorted(os.listdir(image_folder)):
        if fname.endswith(".png"):
            img_path = os.path.join(image_folder, fname)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(TARGET_SIZE)
            img_array = np.array(img) / 255.0  # normalize
            image_list.append(img_array)

    return np.array(image_list, dtype=np.float32)

def save_class_data(label):
    input_folder = os.path.join(IMAGE_DIR, label)
    output_path = os.path.join(OUTPUT_DIR, f"{label}.npy")
    data = images_to_array(input_folder)
    np.save(output_path, data)
    print(f"Saved {label}.npy with shape {data.shape}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_class_data("positive")
    save_class_data("negative")
