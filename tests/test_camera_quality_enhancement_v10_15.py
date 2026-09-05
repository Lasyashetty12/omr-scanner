from pathlib import Path

import cv2
import numpy as np

from omr_preprocess.document_mode import prepare_omr_document_mode

ROOT = Path(__file__).resolve().parents[1]


def test_camera_sharpness_is_not_a_hard_900_gate():
    source = (ROOT / "scanner.py").read_text(encoding="utf-8")

    assert "camera_min_sharpness = 900.0" not in source
    assert (
        "Camera image is not sharp enough for reliable bubble detection"
        not in source
    )
    assert "continue_with_document_mode_enhancement" in source


def test_low_quality_camera_continues_with_enhancement():
    source = (ROOT / "scanner.py").read_text(encoding="utf-8")

    assert "camera_quality_needs_help" in source

    # scanner.py intentionally formats dict indexing across multiple lines.
    # Verify the behaviour without requiring one exact source line.
    assert '"camera_quality_action"' in source
    assert "document_quality[" in source

    assert '"continue_with_document_mode_enhancement"' in source
    assert '"hard_sharpness_gate":' in source


def test_document_mode_has_soft_image_and_brightness_enhancement():
    source = (
        ROOT / "omr_preprocess" / "document_mode.py"
    ).read_text(encoding="utf-8")

    assert "_gentle_illumination_correction" in source
    assert "_lift_paper_whites" in source
    assert "controlled_sharpening" in source
    assert "< 900.0" in source


def test_dark_soft_document_is_enhanced_without_geometry_change():
    image = np.full((800, 600, 3), 105, dtype=np.uint8)

    cv2.rectangle(
        image,
        (30, 30),
        (570, 770),
        (45, 45, 45),
        2,
    )

    for y in range(120, 700, 45):
        for x in (180, 220, 260, 300):
            cv2.circle(
                image,
                (x, y),
                7,
                (55, 55, 55),
                1,
            )

    cv2.circle(
        image,
        (220, 300),
        6,
        (20, 20, 20),
        -1,
    )

    preview, recognition, debug = prepare_omr_document_mode(image)

    assert preview.shape == image.shape
    assert recognition.shape == image.shape
    assert debug["geometry_changed"] is False
    assert debug["recognition_image_modified"] is True


def test_quality_module_keeps_poor_images_non_blocking():
    source = (
        ROOT / "omr_preprocess" / "quality.py"
    ).read_text(encoding="utf-8")

    assert 'can_scan = classification != "REJECT"' in source


def test_existing_v10_14_and_jee_paths_remain():
    scanner = (ROOT / "scanner.py").read_text(encoding="utf-8")
    identity = (ROOT / "identity_reader.py").read_text(encoding="utf-8")

    assert "neet_kcet_reference_orientation_v10_14" in scanner
    assert "jee_identity_corrected_retry_v10_14" in scanner
    assert "jee_roll_ml_disk_v10_14" in identity
    assert "jee_ml_hybrid_gate_v10_11" in scanner
    assert "merge_jee_camera_numerical_records(" in scanner


def test_batch_limit_remains_500():
    source = (
        ROOT / "static" / "app.js"
    ).read_text(encoding="utf-8")

    assert "MAX_BATCH_OMR_FILES = 500" in source
