from pathlib import Path
import ast
import json

import numpy as np

from ml_omr import json_anchor_reader

ROOT = Path(__file__).resolve().parents[1]


def test_kcet_template_is_240_questions_in_four_60_row_columns():
    template = json.loads(
        (
            ROOT
            / "templates"
            / "kcet.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert int(template["total_questions"]) == 240
    assert int(template["questions_per_column"]) == 60
    assert len(template["columns"]) == 4
    assert len(template["question_y_positions"]) == 60


def test_json_coordinate_builder_uses_60_row_column_boundaries():
    template = {
        "total_questions": 240,
        "questions_per_column": 60,
        "options": ["A", "B", "C", "D"],
        "question_y_positions": list(range(100, 160)),
        "columns": [
            {"A": 10, "B": 20, "C": 30, "D": 40},
            {"A": 110, "B": 120, "C": 130, "D": 140},
            {"A": 210, "B": 220, "C": 230, "D": 240},
            {"A": 310, "B": 320, "C": 330, "D": 340},
        ],
    }

    coordinates = (
        json_anchor_reader
        .build_template_json_coordinates(
            template
        )
    )

    assert len(coordinates) == 240

    assert coordinates[60]["A"] == (10.0, 159.0)
    assert coordinates[61]["A"] == (110.0, 100.0)
    assert coordinates[120]["A"] == (110.0, 159.0)
    assert coordinates[121]["A"] == (210.0, 100.0)
    assert coordinates[180]["A"] == (210.0, 159.0)
    assert coordinates[181]["A"] == (310.0, 100.0)
    assert coordinates[240]["A"] == (310.0, 159.0)


def test_json_anchor_reader_forwards_template_row_count_to_both_ml_passes(
    monkeypatch,
):
    seen = []

    def fake_scan_answers_ml(
        *,
        gray,
        coordinates,
        crop_radius,
        questions_per_column,
        **kwargs,
    ):
        seen.append(
            int(
                questions_per_column
            )
        )

        return (
            {1: None},
            {1: {}},
        )

    monkeypatch.setattr(
        json_anchor_reader,
        "scan_answers_ml",
        fake_scan_answers_ml,
    )

    monkeypatch.setattr(
        json_anchor_reader,
        "build_template_json_coordinates",
        lambda _template:
            {
                1: {
                    "A": (10.0, 10.0),
                    "B": (20.0, 10.0),
                    "C": (30.0, 10.0),
                    "D": (40.0, 10.0),
                }
            },
    )

    monkeypatch.setattr(
        json_anchor_reader,
        "_stabilize_json_coordinates",
        lambda json_coordinates, fitted_coordinates, template:
            (
                json_coordinates,
                {},
            ),
    )

    json_anchor_reader.scan_answers_json_anchored(
        gray=np.full(
            (80, 80),
            255,
            dtype=np.uint8,
        ),
        template={
            "questions_per_column": 60,
        },
        fitted_coordinates={},
        crop_radius=12,
    )

    assert seen == [60, 60]


def test_jee_1000_row_sentinel_still_exists():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

    found = False

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if isinstance(
            node.func,
            ast.Name,
        ):
            name = node.func.id
        elif isinstance(
            node.func,
            ast.Attribute,
        ):
            name = node.func.attr
        else:
            name = None

        if name != "scan_answers_ml":
            continue

        for keyword in node.keywords:
            if (
                keyword.arg
                == "questions_per_column"
                and isinstance(
                    keyword.value,
                    ast.Constant,
                )
                and keyword.value.value
                == 1000
            ):
                found = True

    assert found is True
