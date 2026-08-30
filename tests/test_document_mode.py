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
    assert debug["recognition_source"] == "shadow_normalized_grayscale_document"

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

