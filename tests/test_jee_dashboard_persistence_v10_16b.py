from pathlib import Path

from scorer import calculate_jee_score

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_jee_class_supports_lt_in_ui_and_server():
    index = read("static/index.html")
    dashboard = read("static/dashboard.html")
    app = read("app.py")

    assert '<option value="LT">LT</option>' in index
    assert '<option value="LT">LT</option>' in dashboard
    assert '{"11", "12", "LT"}' in app


def test_new_result_prefers_durable_database_id():
    source = read("static/app.js")

    assert "data?.id || data?.scan_id || null" in source
    assert "lastResult?.id" in source
    assert "lastResult?.scan_id" in source


def test_dashboard_prefers_database_id_and_avoids_cached_empty_list():
    source = read("static/dashboard.js")

    assert "row.id || row.scan_id" in source
    assert 'cache: "no-store"' in source
    assert "Date.now()" in source


def test_individual_result_retries_before_not_found():
    source = read("static/result.js")

    assert "RESULT_FETCH_DELAYS_MS" in source
    assert 'cache: "no-store"' in source
    assert "fetchResultWithRetry" in source
    assert "result.exam_date" in source
    assert "result.session" in source


def test_existing_student_metadata_is_refreshed_for_same_roll():
    source = read("database.py")

    assert "_update_existing_student_v10_16" in source
    assert 'method="PATCH"' in source
    assert '"class_name"' in source
    assert '"section"' in source


def test_database_diagnostics_endpoint_exists_without_exposing_keys():
    app = read("app.py")
    database = read("database.py")

    assert '@app.get("/api/storage-status")' in app
    assert "get_database_diagnostics" in app
    assert "def get_database_diagnostics" in database
    assert '"last_error"' in database


def test_scan_response_reports_database_persistence():
    source = read("app.py")

    assert 'result["database_saved"] = True' in source
    assert 'result["database_saved"] = False' in source


def test_cloudinary_artifacts_are_still_saved():
    source = read("app.py")

    assert "upload_scan_images(" in source
    assert 'result["original_image_url"]' in source
    assert 'result["corrected_image_url"]' in source
    assert 'result["bubble_debug_image_url"]' in source
    assert "upload_evaluation_json(" in source


def test_jee_main_marking_plus4_minus1_zero():
    result = calculate_jee_score(
        detected_mcq={
            1: {"answer": "A"},
            2: {"answer": "B"},
            3: {"answer": "BLANK"},
        },
        detected_numerical={
            21: {"answer": "2.0"},
            22: {"answer": "5"},
            23: {"answer": "BLANK"},
        },
        mcq_answer_key={
            "1": "A",
            "2": "A",
            "3": "C",
        },
        numerical_answer_key={
            "21": "2",
            "22": "4",
            "23": "7",
        },
        marking={
            "mcq_correct": 4,
            "mcq_wrong": -1,
            "mcq_blank": 0,
            "mcq_multiple": -1,
            "numerical_correct": 4,
            "numerical_wrong": -1,
            "numerical_blank": 0,
        },
    )

    assert result["score"] == 6
    assert result["correct"] == 2
    assert result["wrong"] == 2
    assert result["blank"] == 2


def test_batch_limit_remains_500():
    source = read("static/app.js")
    assert "MAX_BATCH_OMR_FILES = 500" in source
