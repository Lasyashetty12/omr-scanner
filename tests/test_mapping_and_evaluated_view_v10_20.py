from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_neet_kcet_answer_mapping_uses_proven_fitted_grid():
    source = (ROOT / "scanner.py").read_text(encoding="utf-8")
    assert "_stable_neet_kcet_mapping_v10_20" in source
    assert "scan_answers_ml(" in source
    assert "coordinates=fitted_coordinates" in source
    assert "filled_confidence=float(" in source
    assert "ambiguous_confidence=float(" in source
    assert "questions_per_column=int(" in source


def test_json_ml_identity_recovery_remains():
    source = (ROOT / "scanner.py").read_text(encoding="utf-8")
    assert "recover_identity_choices_ml(" in source


def test_evaluated_omr_has_durable_database_fallback():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    database = (ROOT / "database.py").read_text(encoding="utf-8")

    assert "encode_evaluated_preview_data_url(" in app
    assert '"bubble_debug_image_data_url"' in app
    assert '"database_inline_fallback"' in app
    assert '"bubble_debug_image_data_url"' in database
    assert '"bubble_debug_storage"' in database


def test_result_view_shows_remote_or_inline_evaluated_omr():
    result_js = (ROOT / "static" / "result.js").read_text(encoding="utf-8")
    result_html = (ROOT / "static" / "result.html").read_text(encoding="utf-8")

    assert "bubble_debug_image_url" in result_js
    assert "bubble_debug_image_data_url" in result_js
    assert "inlineFallbackUsed" in result_js
    assert "showEvaluatedOmr" in result_js
    assert 'id="bubbleAnalysisCard"' in result_html
    assert 'id="bubbleDebugPreview"' in result_html


def test_jee_reader_and_scoring_paths_untouched():
    scanner = (ROOT / "scanner.py").read_text(encoding="utf-8")
    app = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "jee_ml_hybrid_gate_v10_11" in scanner
    assert "merge_jee_camera_numerical_records(" in scanner
    assert "calculate_jee_score(" in app
    assert '"max_score": 300' in app
