import numpy as np

from ml_omr.jee_multiscale_multiple import (
    refine_jee_multiscale_multiples,
)


def _prediction(filled, blank):
    ambiguous = max(0.0, 1.0 - filled - blank)
    return {
        "probabilities": {
            "filled": float(filled),
            "blank": float(blank),
            "ambiguous": float(ambiguous),
        }
    }


def _fixture(stable_answer, existing):
    gray = np.full((260, 320), 225, dtype=np.uint8)
    centers = {
        "A": [55, 130],
        "B": [115, 130],
        "C": [175, 130],
        "D": [235, 130],
    }

    option_data = {}
    for option, center in centers.items():
        filled, blank = existing[option]
        option_data[option] = {
            "crop_center": center,
            "ml_filled_probability": float(filled),
            "ml_blank_probability": float(blank),
            "ml_ambiguous_probability": max(
                0.0,
                1.0 - filled - blank,
            ),
        }

    return (
        gray,
        {1: {"answer": stable_answer}},
        {1: stable_answer},
        {
            1: {
                "answer": stable_answer,
                "status": (
                    "answered"
                    if stable_answer in ("A", "B", "C", "D")
                    else "blank"
                ),
                "options": option_data,
            }
        },
    )


def _classifier(by_option):
    sequence = []
    for option in ("A", "B", "C", "D"):
        sequence.extend(by_option[option])

    offset = {"value": 0}

    def classify(crops):
        start = offset["value"]
        end = start + len(crops)
        offset["value"] = end

        return [
            _prediction(filled, blank)
            for filled, blank in sequence[start:end]
        ]

    return classify


def test_one_bad_scale_cannot_create_false_multiple():
    gray, stable, answers, debug = _fixture(
        "A",
        {
            "A": (0.92, 0.04),
            "B": (0.03, 0.94),
            "C": (0.03, 0.94),
            "D": (0.03, 0.94),
        },
    )

    classifier = _classifier(
        {
            "A": [(0.90, 0.05), (0.89, 0.05), (0.88, 0.06)],
            # one crop is a false filled prediction; the other views are blank
            "B": [(0.91, 0.04), (0.07, 0.88), (0.05, 0.92)],
            "C": [(0.04, 0.93), (0.05, 0.92), (0.04, 0.93)],
            "D": [(0.03, 0.94), (0.04, 0.93), (0.03, 0.94)],
        }
    )

    answers, debug = refine_jee_multiscale_multiples(
        gray=gray,
        stable_mcq=stable,
        ml_answers=answers,
        ml_debug=debug,
        _classifier=classifier,
    )

    assert answers[1] == "A"
    assert not debug[1].get(
        "jee_multiscale_ml_multiple",
        False,
    )


def test_three_filled_candidates_never_create_multiple():
    gray, stable, answers, debug = _fixture(
        "A",
        {
            "A": (0.91, 0.04),
            "B": (0.76, 0.12),
            "C": (0.75, 0.13),
            "D": (0.03, 0.94),
        },
    )

    classifier = _classifier(
        {
            "A": [(0.89, 0.05), (0.90, 0.04), (0.88, 0.05)],
            "B": [(0.78, 0.10), (0.80, 0.09), (0.77, 0.11)],
            "C": [(0.76, 0.12), (0.79, 0.10), (0.78, 0.11)],
            "D": [(0.03, 0.94), (0.04, 0.93), (0.03, 0.94)],
        }
    )

    answers, debug = refine_jee_multiscale_multiples(
        gray=gray,
        stable_mcq=stable,
        ml_answers=answers,
        ml_debug=debug,
        _classifier=classifier,
    )

    assert answers[1] == "A"
    assert not debug[1].get(
        "jee_multiscale_ml_multiple",
        False,
    )


def test_false_stable_single_is_blank_only_when_all_four_are_ml_blank():
    gray, stable, answers, debug = _fixture(
        "B",
        {
            "A": (0.03, 0.94),
            "B": (0.04, 0.93),
            "C": (0.03, 0.94),
            "D": (0.03, 0.94),
        },
    )

    classifier = _classifier(
        {
            option: [
                (0.04, 0.92),
                (0.05, 0.91),
                (0.03, 0.94),
            ]
            for option in ("A", "B", "C", "D")
        }
    )

    answers, debug = refine_jee_multiscale_multiples(
        gray=gray,
        stable_mcq=stable,
        ml_answers=answers,
        ml_debug=debug,
        _classifier=classifier,
    )

    assert answers[1] is None
    assert debug[1]["status"] == "blank"
    assert debug[1][
        "jee_multiscale_ml_blank_veto"
    ] is True
