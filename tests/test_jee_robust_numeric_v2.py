from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_jee_production_uses_robust_numeric_only():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(encoding="utf-8")

    process_start = source.index("def process_omr(")
    start = source.index(
        '    elif exam_name == "JEE":',
        process_start,
    )
    end = source.index("\n    else:", start)
    branch = source[start:end]

    assert "scan_jee_answers(" in branch
    assert "scan_jee_numerical_sections_robust(" in branch
    assert 'answers["numerical"]' in branch
    assert "scan_jee_answers_precise(" not in branch


def test_jee_robust_numeric_uses_dominance_and_one_decimal():
    source = (
        ROOT
        / "jee_reader.py"
    ).read_text(encoding="utf-8")

    start = source.index(
        "def detect_numerical_value_robust("
    )
    end = source.index(
        "\ndef scan_jee_numerical_sections_robust(",
        start,
    )
    body = source[start:end]

    assert "decimal_ranked" in body
    assert "selected_decimal" in body
    assert "decimal_candidates" not in body
    assert 'reader": "jee_robust_grid_reader_v2"' in body


def test_jee_robust_numeric_thresholds():
    template = json.loads(
        (
            ROOT
            / "templates"
            / "jee.json"
        ).read_text(encoding="utf-8")
    )

    assert template["jee_numeric_blank_threshold"] == 0.50
    assert template["jee_numeric_filled_threshold"] == 0.68
    assert template["jee_numeric_minimum_gap"] == 0.07
    assert template["jee_numeric_special_threshold"] == 0.64
    assert template["jee_numeric_relaxed_threshold"] == 0.60
    assert template["jee_numeric_strong_gap"] == 0.10
    assert template["jee_numeric_decimal_gap"] == 0.08
