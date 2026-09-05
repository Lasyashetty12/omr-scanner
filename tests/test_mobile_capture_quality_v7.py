from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_camera_capture_has_strict_server_blur_gate():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(encoding="utf-8")

    assert '== "camera_omr.jpg"' in source

    # v10.15 replaces the old strict server blur rejection with:
    # low-quality metrics -> Document Mode enhancement -> recognition.
    assert "camera_min_sharpness = 900.0" not in source
    assert "camera_min_document_sharpness" not in source
    assert "camera_document_sharpness" in source
    assert "camera_quality_needs_help" in source
    assert "continue_with_document_mode_enhancement" in source
    assert (
        "Camera image is not sharp enough for reliable bubble detection."
        not in source
    )

def test_soft_document_mode_preserves_edges_and_sharpens_moderately():
    source = (
        ROOT
        / "omr_preprocess"
        / "document_mode.py"
    ).read_text(encoding="utf-8")

    assert "soft_input" in source
    assert "denoise_d = 3" in source
    assert "denoise_d = 5" in source
    assert "0.34" in source
    assert "adaptiveThreshold" not in source
