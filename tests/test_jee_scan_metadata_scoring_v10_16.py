from pathlib import Path
import json

from scorer import calculate_jee_score

ROOT = Path(__file__).resolve().parents[1]


def test_jee_2026_marking_penalizes_wrong_mcq_and_wrong_numerical():
    result = calculate_jee_score(
        detected_mcq={1: {"answer": "B"}},
        detected_numerical={21: {"answer": "99"}},
        mcq_answer_key={"1": "A"},
        numerical_answer_key={"21": "12"},
        marking={},
    )

    assert result["score"] == -2
    assert result["wrong"] == 2


def test_jee_correct_and_blank_scoring():
    result = calculate_jee_score(
        detected_mcq={
            1: {"answer": "A"},
            2: {"answer": "BLANK"},
        },
        detected_numerical={
            21: {"answer": "12"},
            22: {"answer": "BLANK"},
        },
        mcq_answer_key={
            "1": "A",
            "2": "B",
        },
        numerical_answer_key={
            "21": "12",
            "22": "7",
        },
        marking={},
    )

    assert result["score"] == 8
    assert result["correct"] == 2
    assert result["blank"] == 2


def test_jee_series_keys_use_negative_one_for_wrong_numerical():
    for series in ("P", "Q", "R", "S"):
        data = json.loads(
            (
                ROOT
                / "answer_keys"
                / "jee"
                / f"{series}.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        assert data["marking"]["numerical_wrong"] == -1


def test_scanner_page_has_jee_metadata_controls():
    source = (
        ROOT
        / "static"
        / "index.html"
    ).read_text(
        encoding="utf-8"
    )

    for element_id in (
        "jeeMetadataSection",
        "jeeClass",
        "jeeSection",
        "jeeExamDate",
        "jeeSession",
    ):
        assert f'id="{element_id}"' in source


def test_frontend_sends_jee_metadata_for_single_and_batch_scans():
    source = (
        ROOT
        / "static"
        / "app.js"
    ).read_text(
        encoding="utf-8"
    )

    assert "appendJeeScanMetadata(" in source
    assert '"class_name"' in source
    assert '"section"' in source
    assert '"exam_date"' in source
    assert '"session"' in source
    assert "validateJeeScanMetadata(" in source


def test_scan_api_accepts_and_requires_jee_metadata():
    source = (
        ROOT
        / "app.py"
    ).read_text(
        encoding="utf-8"
    )

    assert 'class_name: str = Form("")' in source
    assert 'section: str = Form("")' in source
    assert 'exam_date: str = Form("")' in source
    assert 'session: str = Form("")' in source

    # v10.16b formats validation expressions over multiple lines.
    assert 'if exam == "jee":' in source

    # app.py formats the allowed Class set over multiple lines.
    # Verify the actual validation semantically instead of requiring
    # one exact source-line representation.
    import re

    assert re.search(
        r'class_name\s+not\s+in\s+\{\s*"11"\s*,\s*"12"\s*,\s*"LT"\s*\}',
        source,
    )

    assert "section" in source
    assert '{"A", "B", "C"}' in source

    assert "session" in source
    assert '{"Morning", "Afternoon"}' in source

    assert "exam_date" in source
    assert "datetime.strptime" in source
    assert '"%Y-%m-%d"' in source

    assert "JEE scan requires Class, Section" in source
    assert "Exam Date and Session." in source

def test_jee_result_has_pcm_and_marking_scheme_metadata():
    source = (
        ROOT
        / "app.py"
    ).read_text(
        encoding="utf-8"
    )

    assert '"stream": "PCM"' in source
    assert '"max_score": 300' in source
    assert '"JEE Main 2026 Paper 1"' in source


def test_database_uses_selected_exam_date_and_session():
    source = (
        ROOT
        / "database.py"
    ).read_text(
        encoding="utf-8"
    )

    assert 'result_data.get("session")' in source
    assert 'result_data.get("exam_date")' in source
    assert '"exam_date": exam_date' in source
    assert '"session": exam_session' in source


def test_dashboard_exposes_selected_exam_date_and_session():
    html = (
        ROOT
        / "static"
        / "dashboard.html"
    ).read_text(
        encoding="utf-8"
    )

    js = (
        ROOT
        / "static"
        / "dashboard.js"
    ).read_text(
        encoding="utf-8"
    )

    assert "<th>Session</th>" in html
    assert "row.session" in js
    assert "row.exam_date" in js


def test_db_list_prefers_per_scan_class_and_section():
    source = (
        ROOT
        / "database.py"
    ).read_text(
        encoding="utf-8"
    )

    assert 'raw_data.get("class")' in source
    assert 'raw_data.get("section")' in source


def test_existing_recognition_contracts_remain():
    scanner = (
        ROOT
        / "scanner.py"
    ).read_text(
        encoding="utf-8"
    )

    identity = (
        ROOT
        / "identity_reader.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "jee_ml_hybrid_gate_v10_11" in scanner
    assert "merge_jee_camera_numerical_records(" in scanner
    assert "jee_identity_corrected_retry_v10_14" in scanner
    assert "jee_roll_ml_disk_v10_14" in identity
    assert "neet_kcet_reference_orientation_v10_14" in scanner


def test_batch_limit_remains_500():
    source = (
        ROOT
        / "static"
        / "app.js"
    ).read_text(
        encoding="utf-8"
    )

    assert "MAX_BATCH_OMR_FILES = 500" in source
