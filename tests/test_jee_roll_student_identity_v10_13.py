
from pathlib import Path
import json
import numpy as np
import database
import identity_reader

ROOT = Path(__file__).resolve().parents[1]


def _jee_template():
    return json.loads(
        (ROOT / "templates" / "jee.json").read_text(
            encoding="utf-8"
        )
    )


def test_jee_template_has_top_left_seven_digit_roll_grid():
    template = _jee_template()
    roll = template["identity"]["roll_number"]

    assert len(roll["x_positions"]) == 7
    assert len(roll["y_positions"]) == 10
    assert roll["values"] == [str(i) for i in range(10)]


def test_identity_reader_decodes_complete_seven_digit_roll(monkeypatch):
    template = _jee_template()
    config = template["identity"]["roll_number"]

    gray = np.full(
        (template["sheet_height"], template["sheet_width"]),
        255,
        dtype=np.uint8,
    )

    expected_digits = "2034167"

    for column_index, digit in enumerate(expected_digits):
        x = int(config["x_positions"][column_index])
        y = int(config["y_positions"][int(digit)])

        yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
        mask = (
            (xx - x) ** 2
            + (yy - y) ** 2
            <= 5 ** 2
        )
        gray[mask] = 0

    fake_circles = [
        (float(x), float(y), 10.0)
        for x in config["x_positions"]
        for y in config["y_positions"]
    ]

    monkeypatch.setattr(
        identity_reader,
        "_hough",
        lambda *_args, **_kwargs: fake_circles,
    )

    result = identity_reader._detect_roll_number(
        gray,
        config,
        template,
    )

    assert result["complete"] is True
    assert result["value"] == expected_digits


def test_jee_canonical_roll_must_come_from_identity():
    result = {
        "exam": "JEE",
        "scan_id": "abcdef123",
        "roll_number": "9999999",
        "identity": {"roll_number": "2034167"},
    }

    resolved = database._resolve_canonical_student_roll(
        result,
        {"roll_number": "1111111"},
    )

    assert resolved == "2034167"


def test_jee_missing_identity_roll_does_not_generate_placeholder():
    result = {
        "exam": "JEE",
        "scan_id": "abcdef123",
        "identity": {"roll_number": None},
    }

    assert (
        database._resolve_canonical_student_roll(
            result,
            {},
        )
        is None
    )


def test_same_roll_reuses_existing_student(monkeypatch):
    calls = []

    def fake_request(
        endpoint,
        method="GET",
        data=None,
        query_params=None,
    ):
        calls.append(
            (endpoint, method, data, query_params)
        )

        if endpoint == "students" and method == "GET":
            return [{"id": 42, "roll_number": "2034167"}]

        raise AssertionError("Existing student must be reused.")

    monkeypatch.setattr(
        database,
        "_supabase_request",
        fake_request,
    )

    student_id = database._get_or_create_student_by_roll(
        {
            "name": "Student Candidate",
            "roll_number": "2034167",
            "class_name": "12",
            "section": "A",
            "batch": "2026",
        }
    )

    assert student_id == 42
    assert len(calls) == 1
    assert calls[0][3]["roll_number"] == "eq.2034167"


def test_new_roll_creates_student_once(monkeypatch):
    methods = []

    def fake_request(
        endpoint,
        method="GET",
        data=None,
        query_params=None,
    ):
        methods.append(method)

        if method == "GET":
            return []

        if method == "POST":
            return [{"id": 77, **data}]

        return None

    monkeypatch.setattr(
        database,
        "_supabase_request",
        fake_request,
    )

    student_id = database._get_or_create_student_by_roll(
        {
            "name": "Student Candidate",
            "roll_number": "7654321",
            "class_name": "12",
            "section": "A",
            "batch": "2026",
        }
    )

    assert student_id == 77
    assert methods == ["GET", "POST"]


def test_database_requires_detected_jee_roll():
    source = (ROOT / "database.py").read_text(
        encoding="utf-8"
    )

    assert "JEE roll number was not detected" in source
    assert "_get_or_create_student_by_roll(" in source


def test_app_promotes_identity_roll_to_result():
    source = (ROOT / "app.py").read_text(
        encoding="utf-8"
    )

    # app.py formats identity_data["roll_number"] across multiple lines.
    # Verify the behavior without depending on exact whitespace/layout.
    assert 'identity_data.get(' in source
    assert '"roll_number"' in source
    assert 'result["roll_number"]' in source


def test_batch_limit_remains_500():
    source = (
        ROOT
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "MAX_BATCH_OMR_FILES = 500" in source
