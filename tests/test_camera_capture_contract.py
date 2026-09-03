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


def test_autocapture_is_immediate_after_four_corner_blocks():
    source = _source()

    assert "const AUTO_CAPTURE_STABLE_CHECKS = 1;" in source
    assert "const AUTO_CAPTURE_CHECK_INTERVAL_MS = 45;" in source
    assert "const analysisWidth = Math.min(640, videoWidth);" in source
    assert "findSolidMarkerInZone(" in source
    assert "findSolidSquareByContrast(" in source
    assert "captureCameraImage(true);" in source


def test_autocapture_readiness_is_corner_only():
    source = _source()

    start = source.index(
        "function isReadyForAutoCapture("
    )
    end = source.index(
        "/* ==========================================================",
        start + 1,
    )

    readiness = source[start:end]

    assert "detection.markerCount === 4" in readiness
    assert "ready: true" in readiness

    assert "isCompleteSheetInFrame(" not in readiness
    assert "isSheetLargeEnough(" not in readiness
    assert "isSheetReasonablyAligned(" not in readiness
    assert "hasExcessiveMovement(" not in readiness
    assert "AUTO_CAPTURE_MIN_SHARPNESS" not in readiness
    assert "cameraFocusWarmupUntil" not in readiness


def test_corner_detector_still_rejects_non_corner_bubble_candidates():
    source = _source()

    assert 'const cornerNames = ["TL", "TR", "BR", "BL"]' in source
    assert "outerCornerGeometry" in source
    assert "markerToSheetRatio" in source
    assert "minimumCornerOccupancy" in source
