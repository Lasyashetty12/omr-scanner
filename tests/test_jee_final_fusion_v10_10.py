from pathlib import Path

import numpy as np

import scanner

ROOT = Path(__file__).resolve().parents[1]


def test_mcq_stable_single_is_never_overridden(monkeypatch):
    def fake_robust(_image, _template):
        return {
            1: {
                "answer": "D",
                "highest_score": 0.50,
                "confidence_gap": 0.20,
            }
        }, {}

    monkeypatch.setattr(
        scanner,
        "scan_jee_mcq_sections_robust",
        fake_robust,
    )

    stable = {
        1: {
            "answer": "B",
            "scores": {
                "A": 0.10,
                "B": 0.90,
                "C": 0.10,
                "D": 0.10,
            },
        }
    }

    merged, _debug = (
        scanner.resolve_jee_camera_mcq_ambiguities(
            np.zeros(
                (20, 20, 3),
                dtype=np.uint8,
            ),
            stable,
            {},
        )
    )

    assert merged[1]["answer"] == "B"
    assert (
        merged[1]["camera_resolver"]
        == "stable_v10_6a_kept"
    )


def test_mcq_false_multiple_can_be_rescued(monkeypatch):
    def fake_robust(_image, _template):
        return {
            10: {
                "answer": "C",
                "highest_score": 0.35,
                "confidence_gap": 0.16,
            }
        }, {
            "1": {
                "calibrated": True,
            }
        }

    monkeypatch.setattr(
        scanner,
        "scan_jee_mcq_sections_robust",
        fake_robust,
    )

    stable = {
        10: {
            "answer": "MULTIPLE",
            "scores": {
                "A": 0.77,
                "B": 0.20,
                "C": 0.94,
                "D": 0.18,
            },
        }
    }

    merged, debug = (
        scanner.resolve_jee_camera_mcq_ambiguities(
            np.zeros(
                (20, 20, 3),
                dtype=np.uint8,
            ),
            stable,
            {},
        )
    )

    assert merged[10]["answer"] == "C"
    assert (
        merged[10]["camera_resolver"]
        == "reference_delta_rescue_v10_10"
    )
    assert debug["resolved_questions"] == [10]


def test_mcq_real_multiple_remains_multiple(monkeypatch):
    def fake_robust(_image, _template):
        return {
            20: {
                "answer": "MULTIPLE",
            }
        }, {}

    monkeypatch.setattr(
        scanner,
        "scan_jee_mcq_sections_robust",
        fake_robust,
    )

    stable = {
        20: {
            "answer": "MULTIPLE",
        }
    }

    merged, _debug = (
        scanner.resolve_jee_camera_mcq_ambiguities(
            np.zeros(
                (20, 20, 3),
                dtype=np.uint8,
            ),
            stable,
            {},
        )
    )

    assert merged[20]["answer"] == "MULTIPLE"


def test_numeric_two_reader_consensus_wins():
    primary = {
        21: {
            "answer": "12.3",
        }
    }

    legacy = {
        21: {
            "answer": "45.6",
        }
    }

    raw = {
        21: {
            "answer": "45.6",
        }
    }

    merged, debug = (
        scanner.merge_jee_camera_numerical_records(
            primary,
            legacy,
            raw,
        )
    )

    assert merged[21]["answer"] == "45.6"
    assert (
        merged[21]["numeric_ensemble_consensus"]
        == "45.6"
    )
    assert (
        debug["questions"]["21"]["consensus"]
        == "45.6"
    )


def test_numeric_current_robust_is_kept_without_consensus():
    primary = {
        46: {
            "answer": "-12.4",
        }
    }

    legacy = {
        46: {
            "answer": "8.3",
        }
    }

    raw = {
        46: {
            "answer": "7.2",
        }
    }

    merged, _debug = (
        scanner.merge_jee_camera_numerical_records(
            primary,
            legacy,
            raw,
        )
    )

    assert merged[46]["answer"] == "-12.4"
    assert (
        merged[46]["numeric_ensemble_source"]
        == "robust_recognition"
    )


def test_numeric_corrected_fallback_rescues_uncertain_primary():
    primary = {
        73: {
            "answer": "UNCERTAIN",
        }
    }

    legacy = {
        73: {
            "answer": "BLANK",
        }
    }

    raw = {
        73: {
            "answer": "-23.49",
        }
    }

    merged, _debug = (
        scanner.merge_jee_camera_numerical_records(
            primary,
            legacy,
            raw,
        )
    )

    assert merged[73]["answer"] == "-23.49"
    assert (
        merged[73]["numeric_ensemble_source"]
        == "robust_corrected"
    )


def test_final_camera_branch_uses_stable_mcq_plus_resolver_and_numeric_ensemble():
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
        "scan_jee_mcq_sections(\n"
        "                    camera_mcq_image,\n"
        "                    template,"
        in branch
    )

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


def test_v10_8_threshold_tuning_is_removed():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "def _decide_jee_camera_mcq("
        not in source
    )

    assert (
        "def detect_jee_camera_question_answer("
        not in source
    )

    assert (
        '"jee_camera_relative_decision"'
        not in source
    )


def test_existing_numerical_algorithm_is_not_rewritten():
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
