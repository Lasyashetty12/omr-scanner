from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _scanner_source():
    return (
        ROOT
        / "scanner.py"
    ).read_text(encoding="utf-8")


def _jee_branch():
    source = _scanner_source()

    process_start = source.index(
        "def process_omr("
    )
    source = source[process_start:]

    start = source.index(
        'elif exam_name == "JEE":'
    )
    end = source.index(
        "\n    else:",
        start,
    )

    return source[start:end]


def _camera_helper():
    source = _scanner_source()

    start = source.index(
        "def prepare_jee_camera_mcq_image("
    )
    end = source.index(
        "# ============================================================\n"
        "# JEE MCQ SCANNER",
        start,
    )

    return source[start:end]


def test_camera_mcq_restores_earlier_stable_reader():
    branch = _jee_branch()

    assert (
        "scan_jee_answers(\n"
        "                recognition_image,"
        in branch
    )

    assert (
        "scan_jee_mcq_sections(\n"
        "                    camera_mcq_image,"
        in branch
    )

    assert (
        "scan_jee_mcq_sections_robust"
        not in branch
    )


def test_camera_mcq_preprocess_uses_recognition_image_not_corrected():
    branch = _jee_branch()

    assert (
        "prepare_jee_camera_mcq_image(\n"
        "                    recognition_image"
        in branch
    )

    assert (
        "prepare_jee_camera_mcq_image(\n"
        "                    corrected"
        not in branch
    )


def test_camera_mcq_helper_never_changes_geometry():
    helper = _camera_helper()

    forbidden = [
        "cv2.resize(",
        "warpPerspective(",
        "getPerspectiveTransform(",
        "perspectiveTransform(",
        "cv2.rotate(",
        "np.rot90(",
    ]

    for item in forbidden:
        assert item not in helper

    assert "original_shape" in helper
    assert "camera_mcq_gray.shape" in helper
    assert "MORPH_OPEN" in helper


def test_camera_mcq_preprocess_preserves_dimensions_at_runtime():
    import scanner

    image = np.full(
        (420, 300, 3),
        235,
        dtype=np.uint8,
    )

    cv2.circle(
        image,
        (90, 190),
        10,
        (20, 20, 20),
        1,
    )

    cv2.circle(
        image,
        (200, 190),
        9,
        (20, 20, 20),
        -1,
    )

    result = (
        scanner.prepare_jee_camera_mcq_image(
            image
        )
    )

    assert result.shape == image.shape[:2]


def test_numerical_reader_is_untouched():
    branch = _jee_branch()

    assert (
        "scan_jee_numerical_sections_robust(\n"
        "                recognition_image,"
        in branch
    )

    assert (
        'answers["numerical"] = (\n'
        '            robust_numerical\n'
        '        )'
        in branch
    )


def test_batch_and_numerical_versions_remain_present():
    jee_reader = (
        ROOT
        / "jee_reader.py"
    ).read_text(encoding="utf-8")

    app_js = (
        ROOT
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert '"local_grid_affine_v10_2"' in jee_reader
    assert "MAX_BATCH_OMR_FILES = 500" in app_js
