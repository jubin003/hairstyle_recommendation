"""
Crops all images in dataset/raw to face-only using MediaPipe.
Run this BEFORE preprocess.py — it replaces raw images with cropped versions.

Usage:
    pip install mediapipe
    python utils/crop_faces.py   (from backend/ folder)

Compatible with MediaPipe 0.10+ (new Tasks API).
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import RAW_DATA_DIR, CLASSES

# ── Build the FaceDetector (new Tasks API) ──────────────────────────────────
_BASE_OPTIONS = mp_python.BaseOptions(
    model_asset_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "blaze_face_short_range.tflite"          # downloaded automatically below
    )
)

def _get_detector():
    """Download the model file if missing, then return a FaceDetector."""
    model_path = _BASE_OPTIONS.model_asset_path

    if not os.path.exists(model_path):
        print("Downloading BlazeFace model…")
        import urllib.request
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_detector/blaze_face_short_range/float16/1/"
            "blaze_face_short_range.tflite"
        )
        urllib.request.urlretrieve(url, model_path)
        print(f"  Saved to {model_path}")

    options = mp_vision.FaceDetectorOptions(
        base_options=_BASE_OPTIONS,
        min_detection_confidence=0.5
    )
    return mp_vision.FaceDetector.create_from_options(options)


detector = _get_detector()


def crop_face(image_path):
    """
    Returns a face-cropped image (numpy array) or None if no face found.
    Adds 20% padding around the bounding box so hair/jawline are included.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]

    # MediaPipe Tasks expects an mp.Image in RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if not result.detections:
        return None  # no face found — image will be kept as-is

    # Use the first (most prominent) detection
    bbox = result.detections[0].bounding_box   # origin_x, origin_y, width, height (pixels)

    x  = bbox.origin_x
    y  = bbox.origin_y
    bw = bbox.width
    bh = bbox.height

    # Add 20% padding to include hair and chin
    pad_x = int(bw * 0.20)
    pad_y = int(bh * 0.20)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return None

    return cropped


def process_dataset():
    total   = 0
    cropped = 0
    skipped = 0
    no_face = 0

    for cls in CLASSES:
        cls_dir = os.path.join(RAW_DATA_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"  Skipping missing folder: {cls_dir}")
            continue

        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"\n{cls}: {len(files)} images")

        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            total += 1

            face = crop_face(fpath)

            if face is None:
                no_face += 1
                print(f"  No face: {fname}")
                continue

            success = cv2.imwrite(fpath, face)
            if success:
                cropped += 1
            else:
                skipped += 1
                print(f"  Write failed: {fname}")

    print(f"\n{'='*45}")
    print(f"Total images    : {total}")
    print(f"Faces cropped   : {cropped}")
    print(f"No face found   : {no_face}  (kept as-is)")
    print(f"Write errors    : {skipped}")
    print(f"{'='*45}")
    print("Done. Now re-run: python utils/preprocess.py")


if __name__ == "__main__":
    print("Cropping faces from raw dataset using MediaPipe…")
    process_dataset()
    detector.close()