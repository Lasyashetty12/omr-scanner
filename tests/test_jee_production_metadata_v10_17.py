import re
import json
from pathlib import Path

from scorer import calculate_jee_score


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def test_jee_metadata_controls_exist_on_scanner():
    html = read("static/index.html")

    for element_id in (
        'id="jeeMetadataSection"',
        'id="jeeClass"',
        'id="jeeSection"',
        'id="jeeExamDate"',
        'id="jeeSession"',
    ):
        assert element_id in html

    assert '<option value="11">Class 11</option>' in html
    assert '<option value="12">Class 12</option>' in html
    assert '<option value="LT">Long Term (LT)</option>' in html
    # Allow normal HTML formatting such as `selected`, extra attributes,
    # or whitespace/newlines while still requiring a real Morning option.
    assert re.search(
        r'<option\s+[^>]*value=["\']Morning["\'][^>]*>\s*Morning\s*</option>',
        html,
        flags=re.IGNORECASE,
    )
    assert re.search(
        r"<option\s+[^>]*value=[\"']Afternoon[\"'][^>]*>\s*Afternoon\s*</option>",
        html,
        flags=re.IGNORECASE,
    )


def test_jee_metadata_is_required_and_sent_with_scan():
    js = read("static/app.js")
    app = read("app.py")

    assert "validateJeeScanMetadata(" in js
    assert "appendJeeScanMetadata(" in js

    for field in (
        '"class_name"',
        '"section"',
        '"exam_date"',
        '"session"',
    ):
        assert field in js

    assert 'class_name not in {"11", "12", "LT"}' in app
    assert 'section not in {"A", "B", "C"}' in app
    assert 'session not in {"Morning", "Afternoon"}' in app
    assert "JEE scan requires Class, Section" in app


def test_jee_main_2026_scoring_math():
    result = calculate_jee_score(
        detected_mcq={
            1: {"answer": "A"},
            2: {"answer": "A"},
            3: {"answer": "BLANK"},
            4: {"answer": "MULTIPLE"},
        },
        detected_numerical={
            21: {"answer": "3.400"},
            22: {"answer": "8"},
            23: {"answer": "BLANK"},
        },
        mcq_answer_key={
            "1": "A",
            "2": "B",
            "3": "C",
            "4": "D",
        },
        numerical_answer_key={
            "21": "3.4",
            "22": "7",
            "23": "9",
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

    assert result["score"] == 5
    assert result["correct"] == 2
    assert result["wrong"] == 2
    assert result["blank"] == 2
    assert result["multiple"] == 1


def test_jee_result_declares_2026_scheme_and_300_max():
    app = read("app.py")

    assert '"max_score": 300' in app
    assert '"name": "JEE Main 2026 Paper 1"' in app
    assert '"mcq_correct": 4' in app
    assert '"mcq_wrong": -1' in app
    assert '"numerical_correct": 4' in app
    assert '"numerical_wrong": -1' in app


def test_dummy_jee_answer_key_is_not_silent_production_data():
    app = read("app.py")

    assert "answer_key_dummy" in app
    assert "answer_key_mode" in app
    assert "TEST ANSWER KEY" in app
    assert "not stored in the production" in app

    p_key = json.loads(
        (ROOT / "answer_keys" / "jee" / "P.json").read_text(
            encoding="utf-8"
        )
    )

    assert p_key["dummy"] is True
    assert p_key["marking"]["mcq_correct"] == 4
    assert p_key["marking"]["mcq_wrong"] == -1
    assert p_key["marking"]["numerical_correct"] == 4
    assert p_key["marking"]["numerical_wrong"] == -1


def test_real_jee_metadata_and_result_are_persisted_for_dashboard():
    app = read("app.py")
    database = read("database.py")
    dashboard = read("static/dashboard.js")

    assert "save_omr_result_to_db(" in app
    assert 'result["exam_date"] = exam_date' in app
    assert 'result["session"] = session' in app
    assert 'result["section"] = section' in app

    assert '"class_name":' in database
    assert '"section":' in database
    assert '"exam_date": exam_date' in database
    assert '"session": exam_session' in database
    assert '"raw_result_json": json.dumps(result_data)' in database

    assert "row.exam_date" in dashboard
    assert "row.session" in dashboard
    assert "row.roll_number" in dashboard
    assert "row.class" in dashboard
    assert "row.section" in dashboard


def test_teacher_result_reload_preserves_production_metadata():
    database = read("database.py")

    assert '"message": raw_data.get("message")' in database
    assert '"marking_scheme": raw_data.get("marking_scheme")' in database
    assert '"evaluation_status": raw_data.get("evaluation_status")' in database
    assert '"answer_key_mode": raw_data.get("answer_key_mode")' in database
