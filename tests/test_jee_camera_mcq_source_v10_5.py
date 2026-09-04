from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _jee_branch():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(encoding="utf-8")

    process_start = source.index(
        "def process_omr("
    )
    source = source[process_start:]

    start = source.index(
        'elif exam_name == "JEE":'
    )
    end = source.index(
        "\n    else:",
        start,
    )

    return source[start:end]


def test_upload_mcq_keeps_stable_baseline_reader():
    branch = _jee_branch()

    assert (
        "scan_jee_answers(\n"
        "                recognition_image,"
        in branch
    )

    # Robust MCQ is conditional, not unconditional.
    assert "if camera_capture:" in branch


def test_camera_mcq_uses_reference_delta_reader_on_corrected_image():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(encoding="utf-8")

    branch = _jee_branch()

    assert "scan_jee_mcq_sections_robust," in source

    assert (
        "scan_jee_mcq_sections_robust(\n"
        "                    corrected,"
        in branch
    )

    assert 'answers["mcq"] = camera_mcq' in branch
    assert 'answers["_mcq_calibration"]' in branch


def test_numerical_reader_is_completely_unchanged():
    branch = _jee_branch()

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

    assert (
        'answers["numerical"] = (\n'
        '            robust_numerical\n'
        '        )'
        in branch
    )


def test_reference_delta_mcq_implementation_exists():
    source = (
        ROOT
        / "jee_reader.py"
    ).read_text(encoding="utf-8")

    assert '"jee_mcq_reference_delta_v10_4"' in source
    assert "_extra_ink_score(" in source


def test_numerical_local_affine_reader_still_exists():
    source = (
        ROOT
        / "jee_reader.py"
    ).read_text(encoding="utf-8")

    assert '"local_grid_affine_v10_2"' in source
