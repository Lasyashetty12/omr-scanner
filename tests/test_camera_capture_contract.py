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


def test_autocapture_is_fast_but_requires_two_good_frames():
    source = _source()

    assert "const AUTO_CAPTURE_STABLE_CHECKS = 2;" in source
    assert "const AUTO_CAPTURE_CHECK_INTERVAL_MS = 45;" in source
    assert "const analysisWidth = Math.min(640, videoWidth);" in source
    assert "const AUTO_CAPTURE_MIN_SHARPNESS = 650;" in source

    assert "findSolidSquareByContrast(" in source
    assert "setTimeout(() =>" not in source
    assert "captureCameraImage(true);" in source


def test_autocapture_readiness_requires_real_sheet_focus_and_stability():
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
    assert "isCompleteSheetInFrame(" in readiness
    assert "isSheetLargeEnough(" in readiness
    assert "isSheetReasonablyAligned(" in readiness
    assert "hasExcessiveMovement(" in readiness
    assert "AUTO_CAPTURE_MIN_SHARPNESS" in readiness
    assert "waiting for camera focus" in readiness
    assert "ready: true" in readiness


def test_live_detection_attaches_frame_sharpness():
    source = _source()

    assert "function estimateFrameSharpness(" in source
    assert "sharpness = estimateFrameSharpness(" in source
    assert "sharpness," in source
