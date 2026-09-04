from pathlib import Path

import scanner

ROOT = Path(__file__).resolve().parents[1]


def _decide(scores, core):
    template = {
        "options": ["A", "B", "C", "D"],
        "blank_threshold": 0.72,
        "filled_threshold": 0.75,
        "multiple_threshold": 0.75,
    }

    return scanner._decide_jee_camera_mcq(
        scores,
        core,
        template,
    )["answer"]


def test_false_camera_multiple_becomes_dominant_single():
    assert _decide(
        {
            "A": 0.90,
            "B": 0.79,
            "C": 0.31,
            "D": 0.28,
        },
        {
            "A": 0.88,
            "B": 0.39,
            "C": 0.12,
            "D": 0.10,
        },
    ) == "A"


def test_real_multiple_is_still_multiple():
    assert _decide(
        {
            "A": 0.91,
            "B": 0.88,
            "C": 0.30,
            "D": 0.27,
        },
        {
            "A": 0.86,
            "B": 0.82,
            "C": 0.10,
            "D": 0.09,
        },
    ) == "MULTIPLE"


def test_narrow_uncertain_band_is_rescued_only_when_core_is_clear():
    assert _decide(
        {
            "A": 0.74,
            "B": 0.55,
            "C": 0.49,
            "D": 0.44,
        },
        {
            "A": 0.83,
            "B": 0.30,
            "C": 0.24,
            "D": 0.21,
        },
    ) == "A"


def test_ambiguous_uncertain_stays_uncertain():
    assert _decide(
        {
            "A": 0.74,
            "B": 0.71,
            "C": 0.52,
            "D": 0.47,
        },
        {
            "A": 0.62,
            "B": 0.59,
            "C": 0.25,
            "D": 0.23,
        },
    ) == "UNCERTAIN"


def test_blank_stays_blank():
    assert _decide(
        {
            "A": 0.69,
            "B": 0.67,
            "C": 0.64,
            "D": 0.62,
        },
        {
            "A": 0.36,
            "B": 0.34,
            "C": 0.31,
            "D": 0.29,
        },
    ) == "BLANK"


def test_v10_7_core_template_override_is_removed():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "def build_jee_camera_mcq_template("
        not in source
    )

    assert (
        "jee_camera_relative_decision"
        in source
    )

    # v10.6a geometry/sampling remains the base:
    # no camera-specific radius=8 or search=2 override.
    assert (
        '"jee_camera_mcq_core_radius"'
        not in source
    )

    assert (
        '"jee_camera_mcq_search_radius"'
        not in source
    )


def test_camera_only_uses_relative_classifier():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(
        encoding="utf-8"
    )

    process_start = source.index(
        "def process_omr("
    )

    process_source = source[
        process_start:
    ]

    jee_start = process_source.index(
        'elif exam_name == "JEE":'
    )

    jee_end = process_source.index(
        "\n    else:",
        jee_start,
    )

    branch = process_source[
        jee_start:jee_end
    ]

    assert (
        'camera_mcq_template["jee_camera_relative_decision"] = True'
        in branch
    )

    assert (
        "scan_jee_mcq_sections(\n"
        "                    camera_mcq_image,\n"
        "                    camera_mcq_template,"
        in branch
    )

    # Upload/default MCQ still uses scan_jee_answers on recognition_image.
    assert (
        "scan_jee_answers(\n"
        "                recognition_image,"
        in branch
    )

    # Numerical path remains untouched.
    assert (
        "scan_jee_numerical_sections_robust(\n"
        "                recognition_image,"
        in branch
    )


def test_camera_classifier_is_opt_in_inside_mcq_scanner():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        'template.get(\n'
        '                    "jee_camera_relative_decision",'
        in source
    )

    assert (
        "detect_jee_camera_question_answer("
        in source
    )

    assert (
        "detect_question_answer("
        in source
    )


def test_numerical_and_batch_versions_stay_present():
    jee_reader = (
        ROOT
        / "jee_reader.py"
    ).read_text(
        encoding="utf-8"
    )

    app_js = (
        ROOT
        / "static"
        / "app.js"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        '"local_grid_affine_v10_2"'
        in jee_reader
    )

    assert (
        "MAX_BATCH_OMR_FILES = 500"
        in app_js
    )
