from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_precise_numerical_samples_translated_template_centres():
    source = (
        ROOT
        / "jee_precise_reader.py"
    ).read_text(encoding="utf-8")

    start = source.index(
        "def _read_numeric_question("
    )

    end = source.index(
        "\ndef scan_jee_numerical_precise(",
        start,
    )

    body = source[start:end]

    assert (
        "centre = translated_digit_points["
        in body
    )

    assert (
        'sampling_mode": "translated_template_center"'
        in body
    )

    assert (
        "centre = assigned_digit_points["
        not in body
    )


def test_production_still_uses_numeric_hybrid_only():
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

    branch = source[start:end]

    assert "scan_jee_answers(" in branch
    assert "scan_jee_numerical_sections_robust(" in branch
    assert 'answers["numerical"]' in branch

    assert "scan_jee_numerical_precise(" not in branch
    assert "scan_jee_answers_precise(" not in branch
    assert "scan_jee_mcq_precise(" not in branch
