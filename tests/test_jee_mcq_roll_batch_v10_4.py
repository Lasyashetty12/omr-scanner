from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_upper_mcq_uses_reference_delta_not_raw_dark_core():
    source = (ROOT / "jee_reader.py").read_text(encoding="utf-8")

    start = source.index("def _classify_mcq(")
    end = source.index(
        "\ndef scan_jee_mcq_sections_robust(",
        start,
    )

    block = source[start:end]

    assert "_extra_ink_score(" in block
    assert "_get_jee_reference_gray(" in block
    assert '"jee_mcq_reference_delta_v10_4"' in block


def test_numerical_calibration_is_untouched():
    source = (ROOT / "jee_reader.py").read_text(encoding="utf-8")

    assert '"local_grid_affine_v10_2"' in source
    assert "def project_local_x(" in source
    assert "translated_sign_x = project_local_x(" in source


def test_roll_number_uses_solid_core_reader():
    source = (ROOT / "identity_reader.py").read_text(encoding="utf-8")

    assert '"solid_roll_grid_v10_4"' in source
    assert "solid_p90_threshold" in source
    assert "solid_p90_ratio" in source
    assert "solid_minimum_p90_gap" in source


def test_database_prefers_detected_roll_number():
    source = (ROOT / "database.py").read_text(encoding="utf-8")

    assert 'identity_data.get("roll_number")' in source
    assert 'result_data.get("roll_number")' in source


def test_batch_upload_limit_is_500():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    app = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    backend = (ROOT / "app.py").read_text(encoding="utf-8")

    assert 'id="imageUpload"' in html
    assert "multiple hidden" in html
    assert "MAX_BATCH_OMR_FILES = 500" in app
    assert '"/scan-batch"' in app
    assert '@app.post("/scan-batch")' in backend
    assert "len(images) > 500" in backend
