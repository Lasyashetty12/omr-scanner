
from __future__ import annotations

import cv2
import numpy as np


def _stretch(gray, low_p=2.5, high_p=98.0):
    low = float(np.percentile(gray, low_p))
    high = float(np.percentile(gray, high_p))

    if high <= low + 1.0:
        return gray.copy()

    out = (
        (gray.astype(np.float32) - low)
        * (255.0 / (high - low))
    )

    return np.clip(out, 0, 255).astype(np.uint8)


def _scanner_tone_curve(gray):
    """
    Scanner-like tone curve:
      - keeps dark pencil/text dark
      - brightens paper background
      - suppresses middle-gray paper texture
    """
    x = np.arange(256, dtype=np.float32)

    # Piecewise curve, intentionally smooth.
    y = np.empty_like(x)

    dark = x <= 90
    mid = (x > 90) & (x <= 205)
    light = x > 205

    # Keep blacks strong.
    y[dark] = x[dark] * 0.92

    # Increase separation in the text/bubble range.
    y[mid] = 82.8 + (x[mid] - 90.0) * 1.22

    # Push paper rapidly toward white.
    y[light] = 223.1 + (x[light] - 205.0) * 0.64

    y = np.clip(y, 0, 255).astype(np.uint8)

    return cv2.LUT(gray, y)


def create_document_preview(corrected_bgr):
    """
    Stronger Vivo-like DOCUMENT PREVIEW.

    DISPLAY ONLY:
      - recognition image is never changed
      - no geometry changes
      - no binary/adaptive thresholding
    """

    if corrected_bgr is None:
        raise ValueError("Document preview received an empty image.")

    if corrected_bgr.ndim == 2:
        gray = corrected_bgr.copy()
    else:
        gray = cv2.cvtColor(
            corrected_bgr,
            cv2.COLOR_BGR2GRAY,
        )

    # --------------------------------------------------------
    # 1. Remove broad lighting / shadow gradient
    # --------------------------------------------------------
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=38,
        sigmaY=38,
    )

    background = np.maximum(
        background,
        1,
    )

    flat = cv2.divide(
        gray,
        background,
        scale=238,
    )

    # --------------------------------------------------------
    # 2. Controlled global contrast
    # --------------------------------------------------------
    flat = _stretch(
        flat,
        low_p=2.5,
        high_p=98.0,
    )

    # --------------------------------------------------------
    # 3. Suppress phone-camera paper grain
    # --------------------------------------------------------
    flat = cv2.fastNlMeansDenoising(
        flat,
        None,
        h=5,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # --------------------------------------------------------
    # 4. Scanner-like white-paper tone curve
    # --------------------------------------------------------
    flat = _scanner_tone_curve(
        flat
    )

    # --------------------------------------------------------
    # 5. Mild local contrast for tiny letters and bubble rings
    # --------------------------------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=1.08,
        tileGridSize=(10, 10),
    )

    flat = clahe.apply(
        flat
    )

    # --------------------------------------------------------
    # 6. Crisp text without hard thresholding
    # --------------------------------------------------------
    soft = cv2.GaussianBlur(
        flat,
        (0, 0),
        sigmaX=0.85,
        sigmaY=0.85,
    )

    sharp = cv2.addWeighted(
        flat,
        1.42,
        soft,
        -0.42,
        0,
    )

    # Final tiny white-background cleanup.
    sharp = np.where(
        sharp >= 244,
        255,
        sharp,
    ).astype(np.uint8)

    return cv2.cvtColor(
        sharp,
        cv2.COLOR_GRAY2BGR,
    )


def prepare_omr_document_mode(corrected_bgr):
    display_image = create_document_preview(
        corrected_bgr
    )

    # CRITICAL: exact recognition image is preserved.
    recognition_image = corrected_bgr.copy()

    debug = {
        "preview_only": True,
        "recognition_image_modified": False,
        "geometry_changed": False,
        "adaptive_threshold_used_for_recognition": False,
        "preview_profile": "vivo_like_strong",
    }

    return (
        display_image,
        recognition_image,
        debug,
    )
