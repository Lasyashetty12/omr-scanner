from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _process_omr_source():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(encoding="utf-8")

    start = source.index(
        "def process_omr("
    )

    return source[start:]


def _process_omr_jee_branch():
    source = _process_omr_source()

    start = source.index(
        'elif exam_name == "JEE":'
    )

    end = source.index(
        "\n    else:",
        start,
    )

    return source[start:end]


def test_jee_baseline_reader_uses_preprocessed_recognition_image():
    branch = _process_omr_jee_branch()

    assert (
        "scan_jee_answers(\n"
        "                recognition_image,"
        in branch
    )


def test_jee_robust_numerical_override_uses_same_preprocessed_image():
    branch = _process_omr_jee_branch()

    assert (
        "scan_jee_numerical_sections_robust(\n"
        "                recognition_image,"
        in branch
    )

    assert (
        "scan_jee_numerical_sections_robust(\n"
        "                corrected,"
        not in branch
    )


def test_preprocessing_is_not_bypassed_by_numeric_override():
    source = _process_omr_source()

    prep = source.index(
        "document_preview, recognition_image, document_mode_debug"
    )

    jee = source.index(
        'elif exam_name == "JEE":'
    )

    robust = source.index(
        "scan_jee_numerical_sections_robust(",
        jee,
    )

    assert prep < jee < robust
