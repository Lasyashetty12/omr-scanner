from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manual_capture_is_not_registration_gated():
    source = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

    assert """captureButton.addEventListener(
        "click",
        () => captureCameraImage(false)
    );""" in source
    assert "if (automatic && !pageCornersDetected)" in source


def test_autocapture_is_single_fast_marker_check():
    source = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

    assert "const AUTO_CAPTURE_STABLE_CHECKS = 1;" in source
    assert "const AUTO_CAPTURE_CHECK_INTERVAL_MS = 45;" in source
    assert "const analysisWidth = Math.min(480, videoWidth);" in source
    assert "setTimeout(() =>" not in source
