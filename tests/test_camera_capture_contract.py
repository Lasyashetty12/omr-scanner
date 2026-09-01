from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manual_capture_is_not_registration_gated():
    source = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

    assert """captureButton.addEventListener(
        "click",
        () => captureCameraImage(false)
    );""" in source
    assert "if (automatic && !pageCornersDetected)" in source


def test_autocapture_uses_two_fast_marker_checks():
    source = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

    assert "const AUTO_CAPTURE_STABLE_CHECKS = 2;" in source
    assert "const AUTO_CAPTURE_CHECK_INTERVAL_MS = 80;" in source
    assert "const analysisWidth = Math.min(640, videoWidth);" in source
    assert "findSolidSquareByContrast(" in source
    assert "setTimeout(() =>" not in source
