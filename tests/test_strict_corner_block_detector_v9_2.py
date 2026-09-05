from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source():
    return (
        ROOT
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")


def _detect_document_corners_block():
    source = _source()

    start = source.index(
        "function detectDocumentCorners()"
    )

    end = source.index(
        "function monitorCornerBlocks(",
        start,
    )

    return source[start:end]


def test_autocapture_still_fires_on_one_valid_four_block_detection():
    source = _source()

    assert "const AUTO_CAPTURE_STABLE_CHECKS = 1;" in source
    assert "captureCameraImage(true);" in source

    ready_start = source.index(
        "function isReadyForAutoCapture("
    )
    ready_end = source.index(
        "/* ==========================================================",
        ready_start + 1,
    )

    ready = source[ready_start:ready_end]

    assert "detection.markerCount === 4" in ready
    assert "ready: true" in ready

    # Accuracy must come from genuine corner-block classification only.
    # Do not reintroduce focus/movement/tilt/page-margin delays.
    assert "AUTO_CAPTURE_MIN_SHARPNESS" not in ready
    assert "hasExcessiveMovement(" not in ready
    assert "isSheetReasonablyAligned(" not in ready
    assert "isCompleteSheetInFrame(" not in ready


def test_active_corner_detector_never_uses_generic_contrast_fallback():
    detector = _detect_document_corners_block()

    assert "findSolidMarkerInZone(" in detector
    assert "findSolidSquareByContrast(" not in detector
    assert "STRICT REGISTRATION-BLOCK CONTRACT" in detector


def test_registration_marker_shape_is_strict_again():
    source = _source()

    assert "aspect < 0.72 || aspect > 1.38" in source
    assert "fill < 0.82" in source
    assert "minimumCornerOccupancy < 0.65" in source
    assert "averageCornerOccupancy < 0.82" in source

    detector = _detect_document_corners_block()
    assert "findSolidSquareByContrast(" not in detector


def test_four_markers_must_have_consistent_physical_size():
    detector = _detect_document_corners_block()

    assert "> 1.85" in detector
    assert "markerToSheetRatio < 0.005" in detector
    assert "markerToSheetRatio > 0.035" in detector
    assert "outerCornerGeometry" in detector


def test_all_four_real_blocks_remain_mandatory():
    detector = _detect_document_corners_block()

    assert "if (!markers.every(Boolean))" in detector
    assert "markerCount: 4" in detector
