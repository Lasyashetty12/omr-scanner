from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_document_sharpness_is_diagnostic_only():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(encoding="utf-8")

    assert "camera_min_sharpness = 900.0" in source
    assert "camera_document_sharpness" in source
    assert "camera_min_document_sharpness" not in source
    assert "camera_capture" in source
    assert "camera_sharpness" in source
    assert "< camera_min_sharpness" in source


def test_quality_module_keeps_raw_and_canonical_sharpness_separate():
    source = (
        ROOT
        / "omr_preprocess"
        / "quality.py"
    ).read_text(encoding="utf-8")

    assert (
        "original_gray = cv2.cvtColor("
        in source
    )

    assert (
        "canonical_gray = cv2.cvtColor("
        in source
    )

    assert (
        "sharpness = float("
        in source
    )

    assert (
        "document_sharpness = float("
        in source
    )

    assert (
        "cv2.Laplacian(original_gray"
        in source
    )

    assert (
        "cv2.Laplacian(canonical_gray"
        in source
    )
