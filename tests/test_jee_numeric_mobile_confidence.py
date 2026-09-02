from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_jee_numeric_mobile_thresholds_are_not_over_strict():
    template = json.loads(
        (
            ROOT
            / "templates"
            / "jee.json"
        ).read_text(encoding="utf-8")
    )

    assert template["jee_precise_numeric_filled_threshold"] == 0.72
    assert template["jee_precise_numeric_blank_threshold"] == 0.48
    assert template["jee_precise_numeric_minimum_gap"] == 0.06
    assert template["jee_precise_numeric_special_threshold"] == 0.68
    assert template["jee_precise_numeric_local_search_radius"] == 3


def test_jee_numeric_reader_uses_tiny_local_search():
    source = (
        ROOT
        / "jee_precise_reader.py"
    ).read_text(encoding="utf-8")

    assert "def _numeric_local_fill(" in source

    start = source.index(
        "def _read_numeric_question("
    )

    end = source.index(
        "\ndef scan_jee_numerical_precise(",
        start,
    )

    body = source[start:end]

    assert "_numeric_local_fill(" in body
    assert "jee_precise_numeric_local_search_radius" in body
