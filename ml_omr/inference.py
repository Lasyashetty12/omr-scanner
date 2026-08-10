
from pathlib import Path
import json
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "bubble_classifier.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"
IMAGE_SIZE = 48

_model = None
_class_names = None

def load_classifier():
    global _model, _class_names

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}. Train first with: python scripts/train_model.py"
            )
        _model = tf.keras.models.load_model(MODEL_PATH)

    if _class_names is None:
        if CLASS_PATH.exists():
            _class_names = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
        else:
            _class_names = ["ambiguous", "blank", "filled"]

    return _model, _class_names

def preprocess_crop(crop):
    if crop is None or crop.size == 0:
        raise ValueError("Empty bubble crop")

    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Local contrast normalization helps with phone lighting.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    crop = clahe.apply(crop)

    h, w = crop.shape[:2]
    size = max(h, w)
    canvas = np.full((size, size), 255, dtype=np.uint8)
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    canvas[y0:y0+h, x0:x0+w] = crop

    resized = cv2.resize(
        canvas,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    return (resized.astype(np.float32) / 255.0)[..., None]

def classify_batch(crops):
    if not crops:
        return []

    model, class_names = load_classifier()

    batch = np.stack(
        [preprocess_crop(c) for c in crops],
        axis=0,
    )

    predictions = model.predict(batch, verbose=0)

    results = []
    for p in predictions:
        idx = int(np.argmax(p))
        results.append({
            "label": class_names[idx],
            "confidence": float(p[idx]),
            "probabilities": {
                class_names[i]: float(p[i])
                for i in range(len(class_names))
            },
        })

    return results
