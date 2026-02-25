import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras import layers
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

from config import TEST_DIR, IMAGE_SIZE, BATCH_SIZE, MODEL_PATH, CLASS_NAMES_PATH, MODEL_SAVE_DIR


def main():
    model = load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    print(f"Classes: {class_names}")

    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        shuffle=False, label_mode='int'
    )

    normalize = layers.Rescaling(1.0 / 255)
    test_ds = test_ds.map(lambda x, y: (normalize(x), y))

    y_true = np.concatenate([y for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds), axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"Test Accuracy: {np.mean(y_true == y_pred) * 100:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    save_path = os.path.join(MODEL_SAVE_DIR, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()