
import json
from pathlib import Path

import numpy as np

import ml_omr.json_anchor_reader as jar


ROOT = Path(__file__).resolve().parents[1]


def load_template():
    return json.loads(
        (
            ROOT
            / "templates"
            / "kcet.json"
        ).read_text(encoding="utf-8")
    )


def test_json_coordinate_map_uses_template_exactly():
    template = load_template()
    coords = jar.build_template_json_coordinates(template)

    assert len(coords) == 240
    assert coords[1]["A"] == (
        float(template["columns"][0]["A"]),
        float(template["question_y_positions"][0]),
    )
    assert coords[61]["D"] == (
        float(template["columns"][1]["D"]),
        float(template["question_y_positions"][0]),
    )


def test_wild_grid_shift_cannot_move_json_geometry():
    template = load_template()
    coords = jar.build_template_json_coordinates(template)

    fitted = {
        q: {
            option: (point[0] + 100, point[1] - 80)
            for option, point in option_map.items()
        }
        for q, option_map in coords.items()
    }

    stabilized, offsets = jar._stabilize_json_coordinates(
        coords,
        fitted,
        template,
    )

    assert offsets[0]["dx"] == 0.0
    assert offsets[0]["dy"] == 0.0
    assert stabilized[1]["A"] == coords[1]["A"]


def test_json_answer_is_primary_and_fitted_only_rescues_blank(monkeypatch):
    template = load_template()
    calls = []

    def fake_reader(gray, coordinates, crop_radius, questions_per_column=None, **_kwargs):
        calls.append(coordinates)

        if len(calls) == 1:
            return (
                {1: "B", 2: None, 3: "MULTIPLE"},
                {
                    1: {"status": "answered", "disk_gap": 0.2, "top_gap": 40, "best_darkness": 90},
                    2: {"status": "blank"},
                    3: {"status": "multiple"},
                },
            )

        return (
            {1: "C", 2: "D", 3: "A"},
            {
                1: {"status": "answered", "disk_gap": 0.2, "top_gap": 50, "best_darkness": 100},
                2: {"status": "answered", "disk_gap": 0.2, "top_gap": 50, "best_darkness": 100},
                3: {"status": "answered", "disk_gap": 0.2, "top_gap": 50, "best_darkness": 100},
            },
        )

    monkeypatch.setattr(jar, "scan_answers_ml", fake_reader)

    coords = jar.build_template_json_coordinates(template)

    answers, debug = jar.scan_answers_json_anchored(
        np.full(
            (template["sheet_height"], template["sheet_width"]),
            255,
            dtype=np.uint8,
        ),
        template,
        coords,
        12,
    )

    assert answers[1] == "B"
    assert answers[2] == "D"
    assert answers[3] == "MULTIPLE"
    assert debug["_json_anchor"]["template_is_geometry_authority"] is True


def test_scanner_uses_json_ml_reader():
    source = (ROOT / "scanner.py").read_text(encoding="utf-8")

    assert "scan_answers_json_anchored(" in source
    assert "recover_identity_choices_ml(" in source


def test_selected_exam_fallback_is_present():
    source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "selected_exam_fallback_v10_19" in source
    assert 'if exam in {"neet", "kcet"}:' in source


def test_previous_jee_metadata_and_db_flow_remain():
    source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert 'class_name: str = Form("")' in source
    assert 'section: str = Form("")' in source
    assert 'exam_date: str = Form("")' in source
    assert 'session: str = Form("")' in source
    assert "save_omr_result_to_db" in source
