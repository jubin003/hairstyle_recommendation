import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from PIL import Image
import keras

from config import MODEL_PATH, CLASS_NAMES_PATH, IMAGE_SIZE

_model = None
_class_names = None


def _load_model():
    global _model, _class_names
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH) as f:
            _class_names = json.load(f)
        print("Model loaded.")
    return _model, _class_names


def predict_face_shape(image_path: str) -> dict:
    model, class_names = _load_model()

    img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    confidences = {cls: round(float(probs[i]) * 100, 2) for i, cls in enumerate(class_names)}
    predicted_class = max(confidences, key=confidences.get)

    return {
        "face_shape": predicted_class,
        "confidence": confidences[predicted_class],
        "all_scores": confidences
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model/predict.py <image_path>")
    else:
        print(predict_face_shape(sys.argv[1]))