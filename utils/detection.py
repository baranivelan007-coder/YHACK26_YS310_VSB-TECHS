"""
detection.py — model loading, YOLO inference, and PASS/FAIL inspection logic.

Pipeline:  image -> YOLO model -> detections (boxes) -> inspection result
"""

import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Where the trained weights live. Override with the MODEL_PATH env variable.
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("model", "best.pt"))

# DEMO THRESHOLD. A detection below this confidence is ignored.
# 0.25 is Ultralytics' default and is NOT a scientifically tuned value.
# In production you would pick this from the precision/recall curve on the
# validation set (see notebooks/training.ipynb, "Choosing a threshold").
DEFAULT_CONF = 0.25

# Image size used for inference. Must match (or be close to) the training imgsz.
IMG_SIZE = 256

_model = None  # cached model instance (loaded once, reused for every request)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(path: str = MODEL_PATH):
    """Load best.pt once and cache it. Raises a clear error if it is missing."""
    global _model
    if _model is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model weights not found at '{path}'.\n"
                "Train the model with notebooks/training.ipynb, download best.pt, "
                "and place it at model/best.pt."
            )
        from ultralytics import YOLO  # imported here so the app starts even if the import is slow
        _model = YOLO(path)
    return _model


def get_class_names() -> dict:
    """Return {class_id: class_name} from the trained model."""
    return load_model().names


def model_info() -> str:
    """Short human-readable description of the loaded model, for the UI."""
    try:
        names = get_class_names()
        return (
            f"Weights: `{MODEL_PATH}`  |  Classes ({len(names)}): "
            + ", ".join(names.values())
        )
    except Exception as e:  # noqa: BLE001
        return f"Model not loaded: {e}"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def to_rgb_uint8(image) -> np.ndarray:
    """Normalise anything Gradio/OpenCV gives us into an RGB uint8 array."""
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:                       # grayscale -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:                 # RGBA -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_detection(image_rgb, conf: float = DEFAULT_CONF):
    """
    Run YOLO on one RGB image.

    Returns:
        annotated_rgb : image with boxes drawn
        detections    : list of dicts {class, confidence, bbox[x1,y1,x2,y2]}
    """
    model = load_model()
    image_rgb = to_rgb_uint8(image_rgb)

    # Ultralytics expects BGR when given a numpy array (OpenCV convention).
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result = model.predict(bgr, conf=conf, imgsz=IMG_SIZE, verbose=False)[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append({
            "class": result.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 3),
            "bbox": [x1, y1, x2, y2],
        })

    # result.plot() draws boxes + labels and returns a BGR image.
    annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections


# ---------------------------------------------------------------------------
# Inspection logic
# ---------------------------------------------------------------------------

def inspect(image_rgb, conf: float = DEFAULT_CONF) -> dict:
    """
    Full inspection of one image.

    Rule (demo):  no detections above the threshold -> PASS
                  one or more detections           -> FAIL

    A real factory would replace this rule with acceptance criteria
    (e.g. max scratch length, allowed defects per cm², critical classes).
    """
    annotated, detections = run_detection(image_rgb, conf)

    count = len(detections)
    status = "FAIL" if count > 0 else "PASS"

    if count:
        top = max(detections, key=lambda d: d["confidence"])
        primary_class = top["class"]
        max_conf = top["confidence"]
        class_counts = {}
        for d in detections:
            class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1
        breakdown = ", ".join(f"{k} ×{v}" for k, v in class_counts.items())
        explanation = (
            f"{count} defect region(s) detected above confidence {conf:.2f} "
            f"({breakdown}). Highest-confidence defect: {primary_class} "
            f"at {max_conf:.0%}. Surface marked FAIL."
        )
    else:
        primary_class = "none"
        max_conf = 0.0
        explanation = (
            f"No defect regions detected above confidence {conf:.2f}. "
            "Surface marked PASS. Note: a PASS means the model found nothing, "
            "not a guarantee that the surface is perfect."
        )

    return {
        "annotated": annotated,
        "detections": detections,
        "defect_count": count,
        "primary_class": primary_class,
        "max_confidence": max_conf,
        "status": status,
        "explanation": explanation,
    }