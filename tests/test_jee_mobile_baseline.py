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

    return source, source[start:end]


def test_jee_mobile_baseline_uses_orb_ecc_registration():
    source, _ = _production_jee_branch()

    assert 'use_orb=(template_exam_name == "JEE"),' in source
    assert 'use_ecc=(template_exam_name == "JEE"),' in source
    assert "ecc_minimum_score=0.80," in source


def test_jee_mobile_baseline_uses_established_answer_reader():
    _, branch = _production_jee_branch()

    assert "scan_jee_answers(" in branch
    assert "scan_jee_answers_robust(" not in branch
    assert "scan_jee_answers_precise(" not in branch


def test_jee_series_search_tolerates_mobile_header_offset():
    template = json.loads(
        (
            ROOT
            / "templates"
            / "jee.json"
        ).read_text(encoding="utf-8")
    )

    assert template["series"]["search_radius"] == 18
    assert template["series"]["search_step"] == 2