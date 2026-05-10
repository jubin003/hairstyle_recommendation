import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KERAS_BACKEND"] = "torch"

import json
import numpy as np
from PIL import Image
import keras

from config import MODEL_PATH, CLASS_NAMES_PATH, IMAGE_SIZE
from model.mediapipe_analysis import analyze_hair, analyze_advanced_geometry

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
    cnn_confidences = {cls: round(float(probs[i]) * 100, 2) for i, cls in enumerate(class_names)}
            
    predicted_class = max(cnn_confidences, key=cnn_confidences.get)

    # Hair Analysis
    hair_features = analyze_hair(image_path)

    return {
        "face_shape": predicted_class,
        "confidence": cnn_confidences[predicted_class],
        "all_scores": cnn_confidences,
        "cnn_scores": cnn_confidences,
        "geo_scores": {},

        "auto_hair_length": hair_features.get("length", "medium"),
        "auto_hair_type": hair_features.get("hair_type", "any")
    }

def predict_live(front_path: str, left_path: str, right_path: str) -> dict:
    """
    Enhanced prediction for live camera multi-angle capture.
    """
    # Base prediction using the front image
    base_pred = predict_face_shape(front_path)
    
    # Advanced 3D geometry from multi-angle
    adv_geo = analyze_advanced_geometry(front_path, left_path, right_path)
    
    # Merge results
    base_pred["advanced_geometry"] = adv_geo
    return base_pred


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model/predict.py <image_path>")
    else:
        print(predict_face_shape(sys.argv[1]))