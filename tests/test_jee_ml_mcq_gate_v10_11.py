from pathlib import Path

import numpy as np

import scanner

ROOT = Path(__file__).resolve().parents[1]


def _template():
    return {
        "mcq_sections": [
            {
                "start_question": 10,
                "total_questions": 4,
                "option_x": {
                    "A": 100,
                    "B": 124,
                    "C": 148,
                    "D": 172,
                },
                "question_y_positions": [
                    100,
                    130,
                    160,
                    190,
                ],
            }
        ]
    }


def _fake_geometry(_image, _template):
    return {
        10: {
            "option_centres": {
                "A": [101, 100],
                "B": [125, 100],
                "C": [149, 100],
                "D": [173, 100],
            }
        },
        11: {
            "option_centres": {
                "A": [101, 130],
                "B": [125, 130],
                "C": [149, 130],
                "D": [173, 130],
            }
        },
        12: {
            "option_centres": {
                "A": [101, 160],
                "B": [125, 160],
                "C": [149, 160],
                "D": [173, 160],
            }
        },
        13: {
            "option_centres": {
                "A": [101, 190],
                "B": [125, 190],
                "C": [149, 190],
                "D": [173, 190],
            }
        },
    }, {
        "calibrated": True,
    }


def test_false_multiple_becomes_blank_from_ml(monkeypatch):
    monkeypatch.setattr(
        scanner,
        "ensure_ml_model_available",
        lambda: None,
    )

    monkeypatch.setattr(
        scanner,
        "scan_jee_mcq_sections_robust",
        _fake_geometry,
    )

    captured = {}

    def fake_ml(**kwargs):
        captured.update(kwargs)

        return {
            10: None,
        }, {
            10: {
                "status": "blank",
                "multiple_options": [],
                "options": {},
            }
        }

    monkeypatch.setattr(
        scanner,
        "scan_answers_ml",
        fake_ml,
    )

    stable = {
        10: {
            "answer": "MULTIPLE",
            "scores": {
                "A": 0.81,
                "B": 0.78,
                "C": 0.22,
                "D": 0.18,
            },
        }
    }

    merged, debug = (
        scanner.resolve_jee_camera_mcq_ambiguities(
            np.zeros(
                (250, 250, 3),
                dtype=np.uint8,
            ),
            stable,
            _template(),
        )
    )

    assert merged[10]["answer"] == "BLANK"
    assert (
        merged[10]["camera_resolver"]
        == "ml_hybrid_blank_v10_11"
    )
    assert merged[10]["scores"] == {
        "A": 0.0,
        "B": 0.0,
        "C": 0.0,
        "D": 0.0,
    }
    assert debug["false_multiple_to_blank"] == [10]

    # JEE bubbles are close together.  Use a tight 21x21 crop so the
    # neighboring option is not included in the ML crop.
    assert captured["crop_radius"] == 10

    # Do not activate NEET/KCET long-column final-row rescue on JEE.
    assert captured["questions_per_column"] == 1000


def test_stable_single_is_preserved_even_if_ml_disagrees(monkeypatch):
    monkeypatch.setattr(
        scanner,
        "ensure_ml_model_available",
        lambda: None,
    )

    monkeypatch.setattr(
        scanner,
        "scan_jee_mcq_sections_robust",
        _fake_geometry,
    )

    monkeypatch.setattr(
        scanner,
        "scan_answers_ml",
        lambda **_kwargs: (
            {
                11: None,
            },
            {
                11: {
                    "status": "blank",
                    "multiple_options": [],
                }
            },
        ),
    )

    stable = {
        11: {
            "answer": "C",
            "scores": {
                "A": 0.10,
                "B": 0.11,
                "C": 0.94,
                "D": 0.12,
            },
        }
    }

    merged, _debug = (
        scanner.resolve_jee_camera_mcq_ambiguities(
            np.zeros(
                (250, 250, 3),
                dtype=np.uint8,
            ),
            stable,
            _template(),
        )
    )

    assert merged[11]["answer"] == "C"
    assert (
        merged[11]["camera_resolver"]
        == "stable_single_kept_v10_11"
    )


def test_ml_single_rescues_stable_multiple(monkeypatch):
    monkeypatch.setattr(
        scanner,
        "ensure_ml_model_available",
        lambda: None,
    )

    monkeypatch.setattr(
        scanner,
        "scan_jee_mcq_sections_robust",
        _fake_geometry,
    )

    monkeypatch.setattr(
        scanner,
        "scan_answers_ml",
        lambda **_kwargs: (
            {
                12: "B",
            },
            {
                12: {
                    "status": "answered",
                    "best_option": "B",
                    "multiple_options": [],
                }
            },
        ),
    )

    stable = {
        12: {
            "answer": "MULTIPLE",
        }
    }

    merged, _debug = (
        scanner.resolve_jee_camera_mcq_ambiguities(
            np.zeros(
                (250, 250, 3),
                dtype=np.uint8,
            ),
            stable,
            _template(),
        )
    )

    assert merged[12]["answer"] == "B"
    assert (
        merged[12]["camera_resolver"]
        == "ml_hybrid_single_v10_11"
    )


def test_real_ml_multiple_is_kept_and_debug_scores_match(monkeypatch):
    monkeypatch.setattr(
        scanner,
        "ensure_ml_model_available",
        lambda: None,
    )

    monkeypatch.setattr(
        scanner,
        "scan_jee_mcq_sections_robust",
        _fake_geometry,
    )

    monkeypatch.setattr(
        scanner,
        "scan_answers_ml",
        lambda **_kwargs: (
            {
                13: "MULTIPLE",
            },
            {
                13: {
                    "status": "multiple",
                    "multiple_options": [
                        "A",
                        "D",
                    ],
                }
            },
        ),
    )

    stable = {
        13: {
            "answer": "MULTIPLE",
        }
    }

    merged, _debug = (
        scanner.resolve_jee_camera_mcq_ambiguities(
            np.zeros(
                (250, 250, 3),
                dtype=np.uint8,
            ),
            stable,
            _template(),
        )
    )

    assert merged[13]["answer"] == "MULTIPLE"
    assert merged[13]["multiple_options"] == [
        "A",
        "D",
    ]
    assert merged[13]["scores"] == {
        "A": 1.0,
        "B": 0.0,
        "C": 0.0,
        "D": 1.0,
    }


def test_ml_uses_hough_calibrated_centres(monkeypatch):
    monkeypatch.setattr(
        scanner,
        "ensure_ml_model_available",
        lambda: None,
    )

    monkeypatch.setattr(
        scanner,
        "scan_jee_mcq_sections_robust",
        _fake_geometry,
    )

    captured = {}

    def fake_ml(**kwargs):
        captured.update(kwargs)
        return {}, {}

    monkeypatch.setattr(
        scanner,
        "scan_answers_ml",
        fake_ml,
    )

    scanner.resolve_jee_camera_mcq_ambiguities(
        np.zeros(
            (250, 250, 3),
            dtype=np.uint8,
        ),
        {},
        _template(),
    )

    assert (
        captured[
            "coordinates"
        ][10]["A"]
        == (101.0, 100.0)
    )


def test_template_coordinates_are_fallback_when_hough_fails(monkeypatch):
    monkeypatch.setattr(
        scanner,
        "ensure_ml_model_available",
        lambda: None,
    )

    def broken_geometry(
        _image,
        _template,
    ):
        raise ValueError(
            "no circles"
        )

    monkeypatch.setattr(
        scanner,
        "scan_jee_mcq_sections_robust",
        broken_geometry,
    )

    captured = {}

    def fake_ml(**kwargs):
        captured.update(kwargs)
        return {}, {}

    monkeypatch.setattr(
        scanner,
        "scan_answers_ml",
        fake_ml,
    )

    scanner.resolve_jee_camera_mcq_ambiguities(
        np.zeros(
            (250, 250, 3),
            dtype=np.uint8,
        ),
        {},
        _template(),
    )

    assert (
        captured[
            "coordinates"
        ][10]["A"]
        == (100.0, 100.0)
    )


def test_hybrid_reader_has_blank_first_and_strict_multiple_rules():
    source = (
        ROOT
        / "ml_omr"
        / "hybrid_reader.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "Blank row check (MUST RUN FIRST"
        in source
    )

    assert (
        "Keep multiple very strict."
        in source
    )

    assert (
        "disk_dark_ratio"
        in source
    )

    assert (
        "ml_blank_probability"
        in source
    )


def test_camera_branch_still_leaves_numerical_path_unchanged():
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
        "resolve_jee_camera_mcq_ambiguities("
        in branch
    )

    assert (
        "scan_jee_numerical_sections_robust(\n"
        "                recognition_image,"
        in branch
    )

    assert (
        "scan_jee_numerical_sections_robust(\n"
        "                    corrected,"
        in branch
    )

    assert (
        "merge_jee_camera_numerical_records("
        in branch
    )


def test_existing_numeric_algorithms_remain_present():
    jee_reader = (
        ROOT
        / "jee_reader.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        '"local_grid_affine_v10_2"'
        in jee_reader
    )

    assert (
        '"uniform_core_v6"'
        in jee_reader
    )

    assert (
        '"decimal_local_contrast_v6_4"'
        in jee_reader
    )


def test_batch_limit_remains_500():
    app_js = (
        ROOT
        / "static"
        / "app.js"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "MAX_BATCH_OMR_FILES = 500"
        in app_js
    )
