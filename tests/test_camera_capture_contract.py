from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source():
    return (
        ROOT
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")


def test_manual_capture_is_a_true_shutter():
    source = _source()

    expected_listener = (
        'captureButton.addEventListener(\n'
        '        "click",\n'
        '        () => captureCameraImage(false)\n'
        '    );'
    )

    assert expected_listener in source
    assert "if (automatic && !pageCornersDetected)" in source

    capture_start = source.index(
        "function captureCameraImage("
    )
    capture_end = source.index(
        "/* ==========================================================",
        capture_start + 1,
    )

    capture_function = source[
        capture_start:capture_end
    ]

    assert "if (!automatic)" in capture_function
    assert "cancelAnimationFrame(" in capture_function


def test_autocapture_uses_one_fast_valid_marker_frame():
    source = _source()

    assert "const AUTO_CAPTURE_STABLE_CHECKS = 1;" in source
    assert "const AUTO_CAPTURE_CHECK_INTERVAL_MS = 45;" in source
    assert "const analysisWidth = Math.min(480, videoWidth);" in source

    assert "findSolidSquareByContrast(" in source
    assert "setTimeout(() =>" not in source
    assert "captureCameraImage(true);" in source


def test_autocapture_readiness_is_four_marker_gated():
    source = _source()

    start = source.index(
        "function isReadyForAutoCapture("
    )
    end = source.index(
        "/* ==========================================================",
        start + 1,
    )

    readiness = source[start:end]

    assert "detection.markerCount !== 4" in readiness
    assert "ready: true" in readiness

    assert "hasExcessiveMovement(" not in readiness
    assert "isSheetLargeEnough(" not in readiness
    assert "isCompleteSheetInFrame(" not in readiness
    assert "isSheetReasonablyAligned(" not in readiness
