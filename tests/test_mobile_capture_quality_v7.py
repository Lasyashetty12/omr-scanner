from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_camera_capture_has_strict_server_blur_gate():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(encoding="utf-8")

    assert '== "camera_omr.jpg"' in source
    # v8 intentionally raises the live-camera server blur gate.
    assert "camera_min_sharpness = 900.0" in source
    assert "camera_min_document_sharpness = 650.0" in source
    assert "Camera image is not sharp enough for reliable bubble detection." in source


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
