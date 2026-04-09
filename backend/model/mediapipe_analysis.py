"""
MediaPipe Tasks API-based face geometry analysis and OpenCV hair analysis.
Uses FaceLandmarker (Tasks API, requires face_landmarker.task model file).
Hair analysis uses OpenCV + HSV colour segmentation — no extra model needed.
"""

import os
import cv2
import numpy as np

# ── MediaPipe Tasks imports ────────────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    _MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    _MP_AVAILABLE = os.path.isfile(_MODEL_PATH)
    if not _MP_AVAILABLE:
        print(f"[mediapipe_analysis] WARNING: model not found at {_MODEL_PATH}. Geometry analysis disabled.")
except ImportError:
    _MP_AVAILABLE = False
    print("[mediapipe_analysis] WARNING: mediapipe not installed. Geometry analysis disabled.")


# ──────────────────────────────────────────────────────────────────────────
# FACE GEOMETRY
# ──────────────────────────────────────────────────────────────────────────

def analyze_face_geometry(image_path: str) -> dict:
    """
    Returns a score distribution over 5 face shapes using FaceLandmarker.
    Returns empty dict on failure so caller can fall back to CNN-only.
    """
    if not _MP_AVAILABLE:
        return {}

    img = cv2.imread(image_path)
    if img is None:
        return {}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    try:
        options = _mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        with _mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            return {}

        lm = result.face_landmarks[0]

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        # ── Key landmark indices (468-point mesh) ──────────────────────
        top_head   = pt(10)
        chin       = pt(152)
        left_cheek = pt(234)
        right_cheek= pt(454)
        left_jaw   = pt(132)
        right_jaw  = pt(361)
        left_fore  = pt(67)
        right_fore = pt(297)

        face_length     = float(np.linalg.norm(top_head - chin))
        cheek_width     = float(np.linalg.norm(left_cheek - right_cheek))
        jaw_width       = float(np.linalg.norm(left_jaw - right_jaw))
        forehead_width  = float(np.linalg.norm(left_fore - right_fore))

        if cheek_width == 0:
            return {}

        l2w    = face_length / cheek_width        # length : width
        c2j    = cheek_width / jaw_width if jaw_width else 1.0
        f2c    = forehead_width / cheek_width

        # ── Heuristic scoring ──────────────────────────────────────────
        scores = {"oval": 0.0, "round": 0.0, "square": 0.0, "heart": 0.0, "oblong": 0.0}

        # OVAL  – balanced proportions
        if 1.25 < l2w < 1.65:
            scores["oval"] += 0.55
        if 1.0 < c2j < 1.35:
            scores["oval"] += 0.45

        # ROUND – close to 1:1 ratio, soft jaw
        if l2w <= 1.25:
            scores["round"] += 0.55
        if c2j < 1.2:
            scores["round"] += 0.45

        # SQUARE – close to 1:1, but stronger jaw (low c2j)
        if l2w <= 1.25:
            scores["square"] += 0.4
        if c2j < 1.1:
            scores["square"] += 0.6

        # HEART  – wide forehead, narrow chin
        if f2c >= 0.92:
            scores["heart"] += 0.4
        if c2j >= 1.3:
            scores["heart"] += 0.6

        # OBLONG – noticeably longer than wide
        if l2w >= 1.6:
            scores["oblong"] += 0.75
        if 0.9 < c2j < 1.25:
            scores["oblong"] += 0.25

        total = sum(scores.values())
        if total == 0:
            return {}

        return {k: round((v / total) * 100, 2) for k, v in scores.items()}

    except Exception as exc:
        print(f"[analyze_face_geometry] error: {exc}")
        return {}


# ──────────────────────────────────────────────────────────────────────────
# HAIR ANALYSIS  (OpenCV — no extra model needed)
# ──────────────────────────────────────────────────────────────────────────

def _get_hair_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Rough hair mask via HSV range + top-quarter crop.
    Works reasonably for dark and medium-toned hair on neutral backgrounds.
    """
    h, w = img_bgr.shape[:2]
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Broad dark-tone mask (covers most natural hair colours)
    lower_dark = np.array([0,   0,   0])
    upper_dark = np.array([180, 255, 110])
    mask_dark  = cv2.inRange(hsv, lower_dark, upper_dark)

    # Light/golden hair
    lower_light = np.array([15,  20, 130])
    upper_light = np.array([40, 180, 255])
    mask_light  = cv2.inRange(hsv, lower_light, upper_light)

    # Red / auburn hair
    lower_red1 = np.array([0,  50, 50])
    upper_red1 = np.array([10, 255, 180])
    lower_red2 = np.array([165, 50, 50])
    upper_red2 = np.array([180, 255, 180])
    mask_red   = cv2.inRange(hsv, lower_red1, upper_red1) | \
                 cv2.inRange(hsv, lower_red2, upper_red2)

    combined = cv2.bitwise_or(mask_dark, mask_light)
    combined = cv2.bitwise_or(combined,  mask_red)

    # Only look in the top 95% of the image (to mostly capture hair while limiting far background)
    roi_mask = np.zeros_like(combined)
    roi_mask[:int(h * 0.95), :] = 255
    combined = cv2.bitwise_and(combined, roi_mask)

    # Morphological cleanup
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=1)

    return combined


def _face_refs_from_landmarks(image_path: str, img_bgr: np.ndarray):
    """
    Run FaceLandmarker to get ear_y, chin_y, face_xmin, face_xmax.
    Falls back to image-fraction estimates if landmarks unavailable.
    """
    h, w = img_bgr.shape[:2]
    # Default fractions
    ear_y    = int(h * 0.45)
    chin_y   = int(h * 0.65)
    face_xmin= int(w * 0.25)
    face_xmax= int(w * 0.75)

    if not _MP_AVAILABLE:
        return ear_y, chin_y, face_xmin, face_xmax

    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        options = _mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        with _mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result   = landmarker.detect(mp_image)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            ear_y    = int((lm[234].y + lm[454].y) / 2 * h)
            chin_y   = int(lm[152].y * h)
            face_xmin= int(lm[234].x * w)
            face_xmax= int(lm[454].x * w)
    except Exception:
        pass

    return ear_y, chin_y, face_xmin, face_xmax


def analyze_hair(image_path: str) -> dict:
    """
    Returns dict: {"length": "short"|"medium"|"long",
                   "hair_type": "straight"|"wavy"|"curly"}
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"length": "medium", "hair_type": "any"}

    h, w = img.shape[:2]
    hair_mask = _get_hair_mask(img)

    ear_y, chin_y, face_xmin, face_xmax = _face_refs_from_landmarks(image_path, img)

    # ── HAIR LENGTH ────────────────────────────────────────────────────────
    margin       = max(10, int((face_xmax - face_xmin) * 0.15))
    left_x       = max(0,   face_xmin - margin)
    right_x      = min(w-1, face_xmax + margin)
    face_height  = max(1, chin_y - ear_y)

    # Measure lateral hair pixels at different Y slices
    def lateral_px(y):
        if y < 0 or y >= h:
            return 0
        row = hair_mask[y, :]
        return int(np.sum(row[:left_x] > 0) + np.sum(row[right_x:] > 0))

    # Long: significant hair below chin
    long_y     = min(h-1, chin_y + int(face_height * 0.6))
    medium_y   = min(h-1, chin_y + int(face_height * 0.15))

    px_long    = lateral_px(long_y)
    px_medium  = lateral_px(medium_y)
    threshold  = w * 0.04  # ~4 % of image width

    if px_long > threshold:
        length_result = "long"
    elif px_medium > threshold:
        length_result = "medium"
    else:
        length_result = "short"

    # ── HAIR TYPE ──────────────────────────────────────────────────────────
    # Analyse texture in the ROI above the ears (top of head only)
    roi_bottom = max(1, ear_y)
    hair_type_result = "straight"

    if roi_bottom > 10:
        roi_mask_top = hair_mask[:roi_bottom, :]
        roi_gray     = cv2.cvtColor(img[:roi_bottom, :], cv2.COLOR_BGR2GRAY)
        roi_masked   = cv2.bitwise_and(roi_gray, roi_gray,
                                       mask=roi_mask_top.astype(np.uint8))

        hair_px = int(np.sum(roi_mask_top > 0))
        if hair_px > 200:
            edges     = cv2.Canny(roi_masked, 40, 120)
            edge_px   = int(np.sum(edges > 0))
            density   = edge_px / hair_px

            # Thresholds calibrated empirically
            if density > 0.18:
                hair_type_result = "curly"
            elif density > 0.09:
                hair_type_result = "wavy"
            else:
                hair_type_result = "straight"

    return {"length": length_result, "hair_type": hair_type_result}
