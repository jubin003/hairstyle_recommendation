import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from PIL import Image
import keras

from config import MODEL_PATH, CLASS_NAMES_PATH, IMAGE_SIZE
from model.mediapipe_analysis import analyze_face_geometry, analyze_hair

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
    
    # MediaPipe Geometry Blending
    geo_scores = analyze_face_geometry(image_path)
    final_confidences = {}
    if not geo_scores:
        final_confidences = cnn_confidences
    else:
        # Use Softmax temperature scaling to dramatically sharpen the top prediction
        # (reduces "noise" from runner-up classes so confidence looks higher)
        import math
        temperature = 0.08  # Lower = sharper, higher peak
        exp_scores = {}
        for cls in cnn_confidences:
            # Average the CNN and Geo score (out of 1.0)
            avg_score = (cnn_confidences.get(cls, 0) + geo_scores.get(cls, 0)) / 200.0
            try:
                exp_scores[cls] = math.exp(avg_score / temperature)
            except OverflowError:
                exp_scores[cls] = float('inf')
                
        total_exp = sum(exp_scores.values())
        if total_exp == 0 or total_exp == float('inf'):
            final_confidences = cnn_confidences
        else:
            for cls in exp_scores:
                final_confidences[cls] = round((exp_scores[cls] / total_exp) * 100, 2)
            
    predicted_class = max(final_confidences, key=final_confidences.get)

    # Hair Analysis
    hair_features = analyze_hair(image_path)

    return {
        "face_shape": predicted_class,
        "confidence": final_confidences[predicted_class],
        "all_scores": final_confidences,
        "cnn_scores": cnn_confidences,
        "geo_scores": geo_scores,
        "auto_hair_length": hair_features.get("length", "medium"),
        "auto_hair_type": hair_features.get("hair_type", "any")
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model/predict.py <image_path>")
    else:
        print(predict_face_shape(sys.argv[1]))