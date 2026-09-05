import cv2
import numpy as np

from omr_preprocess.document_mode import prepare_omr_document_mode


def test_document_mode_removes_colour_and_shadow_without_geometry_change():
    height, width = 600, 420
    horizontal_shadow = np.tile(
        np.linspace(115, 240, width, dtype=np.uint8),
        (height, 1),
    )
    tinted = cv2.merge(
        (
            horizontal_shadow,
            np.clip(horizontal_shadow + 8, 0, 255).astype(np.uint8),
            np.clip(horizontal_shadow + 18, 0, 255).astype(np.uint8),
        )
    )
    cv2.circle(tinted, (210, 300), 14, (0, 0, 0), -1)

    _preview, recognition, debug = prepare_omr_document_mode(tinted)

    assert recognition.shape == tinted.shape
    assert np.array_equal(recognition[:, :, 0], recognition[:, :, 1])
    assert np.array_equal(recognition[:, :, 1], recognition[:, :, 2])
    assert debug["geometry_changed"] is False
    assert debug["recognition_source"] == "adaptive_capture_enhanced_grayscale_document"

    before_range = float(
        np.percentile(horizontal_shadow, 95) - np.percentile(horizontal_shadow, 5)
    )
    after_gray = recognition[:, :, 0]
    background_mask = np.ones((height, width), dtype=bool)
    background_mask[270:331, 180:241] = False
    after_range = float(
        np.percentile(after_gray[background_mask], 95)
        - np.percentile(after_gray[background_mask], 5)
    )
    assert after_range < before_range * 0.65


def test_dim_low_saturation_capture_is_adaptively_enhanced():
    image = np.full((500, 400, 3), (92, 96, 100), dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (370, 470), (65, 67, 72), 3)
    cv2.rectangle(image, (80, 100), (320, 180), (72, 78, 88), -1)
    cv2.circle(image, (200, 300), 15, (55, 42, 40), -1)

    _preview, recognition, debug = prepare_omr_document_mode(image)
    stats = debug["image_characteristics"]

    assert recognition.shape == image.shape
    assert stats["low_brightness_enhanced"] is True
    assert stats["low_saturation_enhanced"] is True
    assert stats["enhanced_brightness"] > stats["brightness"]
