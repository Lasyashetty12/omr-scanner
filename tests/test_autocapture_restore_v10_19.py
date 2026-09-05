from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source():
    return (
        ROOT
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")


def _block(source, start_token, end_token):
    start = source.index(start_token)
    end = source.index(end_token, start + 1)
    return source[start:end]


def test_autocapture_is_corner_only_and_immediate():
    source = _source()

    assert "const AUTO_CAPTURE_STABLE_CHECKS = 1;" in source
    assert "const AUTO_CAPTURE_CHECK_INTERVAL_MS = 45;" in source

    readiness = _block(
        source,
        "function isReadyForAutoCapture(",
        "/* ==========================================================",
    )

    assert "detection.markerCount === 4" in readiness
    assert "ready: true" in readiness

    for forbidden in (
        "AUTO_CAPTURE_MIN_SHARPNESS",
        "cameraFocusWarmupUntil",
        "hasExcessiveMovement(",
        "isSheetLargeEnough(",
        "isCompleteSheetInFrame(",
        "isSheetReasonablyAligned(",
    ):
        assert forbidden not in readiness


def test_mobile_corner_tolerances_match_working_autocapture():
    source = _source()

    assert "aspect < 0.50 || aspect > 1.90" in source
    assert "fill < 0.65" in source
    assert "> 1.95" in source
    assert "markerToSheetRatio < 0.009" in source
    assert "markerToSheetRatio > 0.038" in source

    assert "minimumCornerOccupancy < 0.65" not in source
    assert "averageCornerOccupancy < 0.82" not in source


def test_monitor_calls_automatic_capture_exactly_once():
    source = _source()

    monitor = _block(
        source,
        "function monitorCornerBlocks(timestamp) {",
        "/* ==========================================================",
    )

    assert monitor.count("captureCameraImage(true);") == 1
    assert "autoCaptureTriggered = true;" in monitor
    assert "isReadyForAutoCapture(" in monitor


def test_detector_still_requires_all_four_real_blocks():
    source = _source()

    detector = _block(
        source,
        "function detectDocumentCorners()",
        "function monitorCornerBlocks(",
    )

    assert 'const cornerNames = ["TL", "TR", "BR", "BL"]' in detector
    assert "if (!markers.every(Boolean))" in detector
    assert "markerCount: 4" in detector
    assert "outerCornerGeometry" in detector
    assert "findSolidMarkerInZone(" in detector
    assert "findSolidSquareByContrast(" not in detector


def test_unrelated_new_jee_metadata_code_is_preserved():
    source = _source()

    for token in (
        "jeeMetadataSection",
        "jeeClass",
        "jeeSection",
        "jeeExamDate",
        "jeeSession",
        "validateJeeScanMetadata(",
        "appendJeeScanMetadata(",
        "MAX_BATCH_OMR_FILES = 500",
    ):
        assert token in source
