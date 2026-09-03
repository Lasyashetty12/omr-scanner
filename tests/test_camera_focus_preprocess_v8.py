from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v9_camera_focus_is_best_effort_not_autocapture_gate():
    source = (
        ROOT
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "const AUTO_CAPTURE_STABLE_CHECKS = 1;" in source

    # Keep camera-quality hints where supported.
    assert 'advanced.focusMode = "continuous"' in source
    assert 'advanced.exposureMode = "continuous"' in source
    assert "const JPEG_QUALITY =\n    0.97;" in source
    assert "ideal:\n                        2560" in source
    assert "ideal:\n                        1440" in source

    # But focus MUST NOT delay corner-block autocapture.
    start = source.index(
        "function isReadyForAutoCapture("
    )
    end = source.index(
        "/* ==========================================================",
        start + 1,
    )
    readiness = source[start:end]

    assert "AUTO_CAPTURE_MIN_SHARPNESS" not in readiness
    assert "cameraFocusWarmupUntil" not in readiness
    assert "previousSharpness" not in readiness


def test_v8_server_rejects_soft_camera_sheet():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(encoding="utf-8")

    assert "camera_min_sharpness = 900.0" in source
    assert "camera_document_sharpness" in source
    assert "camera_min_document_sharpness" not in source
    assert "camera_sharpness" in source
    assert "not sharp enough for reliable bubble detection" in source


def test_v8_soft_preprocessing_extends_to_moderately_soft_images():
    source = (
        ROOT
        / "omr_preprocess"
        / "document_mode.py"
    ).read_text(encoding="utf-8")

    assert "< 900.0" in source
    assert "denoise_d = 3" in source
    assert "0.34" in source
