import numpy as np

from ml_omr.jee_multiscale_multiple import (
    refine_jee_multiscale_multiples,
)


def _prediction(
    filled,
    blank,
):
    ambiguous = max(
        0.0,
        1.0
        - float(filled)
        - float(blank),
    )

    return {
        "label":
            (
                "filled"
                if filled >= blank
                else "blank"
            ),
        "confidence":
            max(
                filled,
                blank,
                ambiguous,
            ),
        "probabilities": {
            "filled":
                float(filled),
            "blank":
                float(blank),
            "ambiguous":
                float(ambiguous),
        },
    }


def _fixture(
    stable_answer,
    existing,
):
    gray = np.full(
        (260, 320),
        225,
        dtype=np.uint8,
    )

    centers = {
        "A": [55, 130],
        "B": [115, 130],
        "C": [175, 130],
        "D": [235, 130],
    }

    options = {}

    for option, center in centers.items():
        filled, blank = existing[
            option
        ]

        options[
            option
        ] = {
            "crop_center":
                center,
            "ml_filled_probability":
                float(filled),
            "ml_blank_probability":
                float(blank),
            "ml_ambiguous_probability":
                max(
                    0.0,
                    1.0
                    - filled
                    - blank,
                ),
        }

    return (
        gray,
        {
            1: {
                "answer":
                    stable_answer,
            }
        },
        {
            1:
                stable_answer
        },
        {
            1: {
                "answer":
                    stable_answer,
                "status":
                    (
                        "answered"
                        if stable_answer
                        in (
                            "A",
                            "B",
                            "C",
                            "D",
                        )
                        else "blank"
                    ),
                "options":
                    options,
            }
        },
    )


def _classifier(
    by_option,
):
    sequence = []

    for option in (
        "A",
        "B",
        "C",
        "D",
    ):
        sequence.extend(
            by_option[
                option
            ]
        )

    offset = {
        "value": 0,
    }

    def classify(
        crops,
    ):
        start = offset[
            "value"
        ]
        end = (
            start
            + len(crops)
        )

        offset[
            "value"
        ] = end

        return [
            _prediction(
                filled,
                blank,
            )
            for filled, blank
            in sequence[
                start:end
            ]
        ]

    return classify


def test_wider_ml_views_recover_second_fill():
    (
        gray,
        stable,
        answers,
        debug,
    ) = _fixture(
        "A",
        {
            "A": (0.92, 0.04),
            "B": (0.03, 0.94),
            # Tight existing crop misses C.
            "C": (0.18, 0.70),
            "D": (0.03, 0.94),
        },
    )

    classify = _classifier(
        {
            "A": [
                (0.90, 0.05),
                (0.88, 0.06),
                (0.86, 0.07),
            ],
            "B": [
                (0.03, 0.94),
                (0.04, 0.93),
                (0.03, 0.94),
            ],
            "C": [
                (0.78, 0.12),
                (0.84, 0.08),
                (0.81, 0.10),
            ],
            "D": [
                (0.04, 0.93),
                (0.03, 0.94),
                (0.04, 0.93),
            ],
        }
    )

    answers, debug = (
        refine_jee_multiscale_multiples(
            gray=gray,
            stable_mcq=stable,
            ml_answers=answers,
            ml_debug=debug,
            _classifier=classify,
        )
    )

    assert answers[1] == "MULTIPLE"

    assert set(
        debug[
            1
        ][
            "multiple_options"
        ]
    ) == {
        "A",
        "C",
    }


def test_empty_options_stay_single():
    (
        gray,
        stable,
        answers,
        debug,
    ) = _fixture(
        "B",
        {
            "A": (0.03, 0.94),
            "B": (0.91, 0.05),
            "C": (0.03, 0.94),
            "D": (0.03, 0.94),
        },
    )

    classify = _classifier(
        {
            "A": [
                (0.05, 0.91),
                (0.04, 0.93),
                (0.05, 0.91),
            ],
            "B": [
                (0.89, 0.06),
                (0.91, 0.05),
                (0.87, 0.07),
            ],
            "C": [
                (0.06, 0.90),
                (0.05, 0.92),
                (0.06, 0.90),
            ],
            "D": [
                (0.05, 0.92),
                (0.04, 0.93),
                (0.05, 0.92),
            ],
        }
    )

    answers, debug = (
        refine_jee_multiscale_multiples(
            gray=gray,
            stable_mcq=stable,
            ml_answers=answers,
            ml_debug=debug,
            _classifier=classify,
        )
    )

    assert answers[1] == "B"

    assert not debug[
        1
    ].get(
        "jee_multiscale_ml_multiple",
        False,
    )


def test_three_supported_options_are_not_forced_multiple():
    (
        gray,
        stable,
        answers,
        debug,
    ) = _fixture(
        "A",
        {
            "A": (0.90, 0.05),
            "B": (0.03, 0.94),
            "C": (0.03, 0.94),
            "D": (0.03, 0.94),
        },
    )

    classify = _classifier(
        {
            "A": [
                (0.88, 0.06),
                (0.90, 0.05),
                (0.87, 0.07),
            ],
            "B": [
                (0.76, 0.13),
                (0.78, 0.12),
                (0.75, 0.14),
            ],
            "C": [
                (0.74, 0.15),
                (0.76, 0.13),
                (0.75, 0.14),
            ],
            "D": [
                (0.03, 0.94),
                (0.04, 0.93),
                (0.03, 0.94),
            ],
        }
    )

    answers, _ = (
        refine_jee_multiscale_multiples(
            gray=gray,
            stable_mcq=stable,
            ml_answers=answers,
            ml_debug=debug,
            _classifier=classify,
        )
    )

    assert answers[1] == "A"


def test_blank_requires_exactly_two_strong_options():
    (
        gray,
        stable,
        answers,
        debug,
    ) = _fixture(
        "BLANK",
        {
            "A": (0.78, 0.12),
            "B": (0.03, 0.94),
            "C": (0.80, 0.10),
            "D": (0.03, 0.94),
        },
    )

    classify = _classifier(
        {
            "A": [
                (0.84, 0.09),
                (0.86, 0.08),
                (0.82, 0.10),
            ],
            "B": [
                (0.04, 0.93),
                (0.03, 0.94),
                (0.04, 0.93),
            ],
            "C": [
                (0.81, 0.11),
                (0.85, 0.08),
                (0.83, 0.09),
            ],
            "D": [
                (0.04, 0.93),
                (0.03, 0.94),
                (0.04, 0.93),
            ],
        }
    )

    answers, debug = (
        refine_jee_multiscale_multiples(
            gray=gray,
            stable_mcq=stable,
            ml_answers=answers,
            ml_debug=debug,
            _classifier=classify,
        )
    )

    assert answers[1] == "MULTIPLE"

    assert set(
        debug[
            1
        ][
            "multiple_options"
        ]
    ) == {
        "A",
        "C",
    }
