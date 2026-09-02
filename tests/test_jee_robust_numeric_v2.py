from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def _production_jee_branch():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(encoding="utf-8")

    process_start = source.index(
        "def process_omr("
    )

    start = source.index(
        '    elif exam_name == "JEE":',
        process_start,
    )

    end = source.index(
        "\n    else:",
        start,
    )

    return source[start:end]


def test_jee_production_still_uses_stable_mcq_plus_robust_numeric():
    branch = _production_jee_branch()

    assert "scan_jee_answers(" in branch
    assert "scan_jee_numerical_sections_robust(" in branch
    assert 'answers["numerical"]' in branch
    assert "scan_jee_answers_precise(" not in branch


def test_legacy_v2_contract_is_superseded_by_solid_core_v5():
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

    assert "_classify_numerical_column(" in body
    assert "filled_candidates" in body
    assert "decimal_status" in body
    assert 'reader":\n            "jee_solid_core_reader_v5"' in body


def test_v5_numeric_controls_exist():
    template = json.loads(
        (
            ROOT
            / "templates"
            / "jee.json"
        ).read_text(encoding="utf-8")
    )

    # v6 intentionally expanded the numerical core from 3 px to 5 px
    # so printed digits are rejected using wider centre uniformity.
    assert template[
        "jee_numeric_solid_core_radius"
    ] == 5

    assert template[
        "jee_numeric_solid_mean_ratio"
    ] == 0.55

    assert template[
        "jee_numeric_solid_spread_ratio"
    ] == 0.35

    assert template[
        "jee_numeric_special_mean_ratio"
    ] == 0.55
