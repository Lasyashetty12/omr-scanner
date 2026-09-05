from pathlib import Path

import app
from fastapi.testclient import TestClient
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_kcet_neet_question_limits_are_explicit():
    assert app.NEET_QUESTION_LIMIT == 180
    assert app.KCET_PCM_QUESTION_LIMIT == 180
    assert app.KCET_PCMB_QUESTION_LIMIT == 240


def test_question_limit_removes_rows_above_selected_exam_limit():
    answers = {
        1: {"answer": "A"},
        "180": {"answer": "B"},
        181: {"answer": "C"},
        "240": {"answer": "D"},
        "invalid": {"answer": "A"},
    }

    limited = app.limit_question_mapping(answers, 180)

    assert limited == {
        1: {"answer": "A"},
        180: {"answer": "B"},
    }


def test_detected_omr_class_is_normalized_for_dashboard_filters():
    assert app.normalize_detected_class("I") == "11"
    assert app.normalize_detected_class("II") == "12"
    assert app.normalize_detected_class("LT") == "LT"
    assert app.normalize_detected_class("unknown") == ""


def test_kcet_neet_ui_collects_only_operator_selected_metadata():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert 'value="kcet_neet"' in html
    assert "Detect from OMR" in html
    assert 'id="kcetNeetExamDate"' in html
    assert 'id="kcetNeetSection"' in html
    assert 'id="kcetNeetSession"' in html
    assert 'id="kcetNeetClass"' not in html


def test_kcet_neet_frontend_submits_metadata_and_pcm_stream():
    source = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

    assert '["kcet", "neet", "kcet_neet"].includes(exam)' in source
    assert 'formData.append(\n        "exam_date"' in source
    assert 'formData.append(\n        "section"' in source
    assert 'formData.append(\n        "session"' in source
    assert 'exam === "kcet" || exam === "kcet_neet"' in source


def test_backend_uses_detected_exam_and_identity_fields():
    source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert 'detected_exam not in {"NEET", "KCET"}' in source
    assert 'exam = detected_exam.lower()' in source
    assert 'result["roll_number"]' in source
    assert 'result["class"] = detected_class' in source
    assert '"section": section' in source
    assert '"exam_date": exam_date' in source
    assert '"session": session' in source


def test_neet_and_kcet_pcm_store_only_180_question_results():
    source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "NEET_QUESTION_LIMIT = 180" in source
    assert "KCET_PCM_QUESTION_LIMIT = 180" in source
    assert "KCET_PCMB_QUESTION_LIMIT = 240" in source
    assert "if int(key) <= NEET_QUESTION_LIMIT" in source
    assert "if int(key) <= question_limit" in source
    assert '"total_questions":\n                    NEET_QUESTION_LIMIT' in source
    assert '"total_questions":\n                    question_limit' in source


@pytest.mark.parametrize(
    ("detected_exam", "stream", "expected_total"),
    [
        ("NEET", "pcmb", 180),
        ("NEET", "pcm", 180),
        ("KCET", "pcm", 180),
        ("KCET", "pcmb", 240),
    ],
)
def test_scan_contract_uses_omr_identity_metadata_and_question_limit(
    monkeypatch,
    tmp_path,
    detected_exam,
    stream,
    expected_total,
):
    saved = {}
    detected_answers = {
        question: {"answer": "A"}
        for question in range(1, 241)
    }

    monkeypatch.setattr(app, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(app, "RESULT_DIR", str(tmp_path / "results"))
    (tmp_path / "uploads").mkdir()
    (tmp_path / "results").mkdir()

    monkeypatch.setattr(
        app,
        "process_omr",
        lambda *_args, **_kwargs: {
            "quality": {},
            "identity": {
                "roll_number": "40316817",
                "class": "II",
                "exam": detected_exam,
            },
            "series": {"value": "Q"},
            "answers": detected_answers,
            "template": {"total_questions": 240},
        },
    )
    monkeypatch.setattr(
        app,
        "load_answer_key_for_exam",
        lambda exam_name, series: {
            "exam": exam_name.upper(),
            "series": series,
            "answers": {
                str(question): "A"
                for question in range(1, 241)
            },
            "marking": {
                "correct": 4 if exam_name == "neet" else 1,
                "wrong": -1 if exam_name == "neet" else 0,
                "blank": 0,
                "multiple": -1 if exam_name == "neet" else 0,
            },
        },
    )
    monkeypatch.setattr(app, "save_debug_images", lambda *_args: None)
    monkeypatch.setattr(app, "cloudinary_enabled", lambda: False)

    def capture_database_save(result, student_info=None):
        saved["result"] = result
        saved["student_info"] = student_info
        return 321

    monkeypatch.setattr(app, "save_omr_result_to_db", capture_database_save)

    response = TestClient(app.app).post(
        "/scan",
        data={
            "exam": "kcet_neet",
            "stream": stream,
            "section": "B",
            "exam_date": "2026-09-05",
            "session": "Afternoon",
        },
        files={
            "image": (
                "omr.jpg",
                b"synthetic-image",
                "image/jpeg",
            ),
        },
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["exam"] == detected_exam
    assert result["detected_exam"] == detected_exam
    assert result["roll_number"] == "40316817"
    assert result["class"] == "12"
    assert result["section"] == "B"
    assert result["exam_date"] == "2026-09-05"
    assert result["session"] == "Afternoon"
    assert result["paper_code"] == "Q"
    assert result["total_questions"] == expected_total
    assert len(result["answers"]) == expected_total
    assert len(result["question_results"]) == expected_total
    assert str(expected_total) in result["answers"]
    assert str(expected_total + 1) not in result["answers"]
    assert result["student"] == {
        "name": "Student Candidate",
        "roll_number": "40316817",
        "class": "12",
        "section": "B",
    }
    assert result["exam_info"] == {
        "exam_type": detected_exam,
        "paper_code": "Q",
        "paper_series": "Q",
        "exam_date": "2026-09-05",
        "session": "Afternoon",
    }
    assert saved["student_info"]["roll_number"] == "40316817"
    assert saved["student_info"]["class_name"] == "12"
    assert saved["student_info"]["section"] == "B"
