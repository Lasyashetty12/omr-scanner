import cv2
import numpy as np

from ml_omr.final_ml_consensus import (
    refine_neet_kcet_answers_with_ml,
    rescue_jee_multiple_from_ml_debug,
)


def _template():
    return {
        "exam_name": "KCET",
        "sheet_width": 420,
        "sheet_height": 280,
        "questions_per_column": 2,
        "options": ["A", "B", "C", "D"],
        "columns": [
            {
                "A": 80,
                "B": 130,
                "C": 180,
                "D": 230,
            }
        ],
        "question_y_positions": [90, 150],
    }


def _sheet(fills):
    template = _template()

    reference = np.full(
        (
            template["sheet_height"],
            template["sheet_width"],
        ),
        235,
        dtype=np.uint8,
    )

    current = reference.copy()
    coordinates = {}

    for question in (1, 2):
        coordinates[question] = {}

        y = template["question_y_positions"][
            question - 1
        ]

        for option, x in template["columns"][0].items():
            coordinates[question][option] = (
                float(x),
                float(y),
            )

            cv2.circle(
                reference,
                (x, y),
                10,
                80,
                2,
                lineType=cv2.LINE_AA,
            )

            cv2.circle(
                current,
                (x, y),
                10,
                80,
                2,
                lineType=cv2.LINE_AA,
            )

    for question, option, value in fills:
        x, y = coordinates[question][option]

        cv2.circle(
            current,
            (int(x), int(y)),
            7,
            int(value),
            -1,
            lineType=cv2.LINE_AA,
        )

    current = cv2.GaussianBlur(
        current,
        (3, 3),
        0,
    )

    return current, reference, coordinates, template


def _debug(coordinates, filled_probabilities):
    result = {}

    for question, option_map in coordinates.items():
        options = {}

        for option, center in option_map.items():
            filled = float(
                filled_probabilities.get(
                    (question, option),
                    0.04,
                )
            )

            options[option] = {
                "crop_center": [
                    int(center[0]),
                    int(center[1]),
                ],
                "ml_filled_probability": filled,
                "ml_blank_probability": max(
                    0.02,
                    0.94 - filled,
                ),
                "ml_ambiguous_probability": 0.02,
            }

        result[question] = {
            "answer": None,
            "status": "blank",
            "options": options,
        }

    return result


def test_empty_printed_rings_override_false_multiple():
    current, reference, coordinates, template = _sheet([])

    answers, debug = refine_neet_kcet_answers_with_ml(
        gray=current,
        coordinates=coordinates,
        template=template,
        raw_answers={
            1: "MULTIPLE",
            2: None,
        },
        ml_debug=_debug(coordinates, {}),
        reference_gray=reference,
    )

    assert answers[1] is None
    assert debug[1]["status"] == "blank"


def test_single_filled_bubble():
    current, reference, coordinates, template = _sheet(
        [(1, "C", 35)]
    )

    answers, _ = refine_neet_kcet_answers_with_ml(
        gray=current,
        coordinates=coordinates,
        template=template,
        raw_answers={
            1: None,
            2: None,
        },
        ml_debug=_debug(
            coordinates,
            {(1, "C"): 0.93},
        ),
        reference_gray=reference,
    )

    assert answers[1] == "C"


def test_unequal_multiple_bubbles():
    current, reference, coordinates, template = _sheet(
        [
            (1, "A", 35),
            (1, "D", 72),
        ]
    )

    answers, debug = refine_neet_kcet_answers_with_ml(
        gray=current,
        coordinates=coordinates,
        template=template,
        raw_answers={
            1: "A",
            2: None,
        },
        ml_debug=_debug(
            coordinates,
            {
                (1, "A"): 0.96,
                (1, "D"): 0.82,
            },
        ),
        reference_gray=reference,
    )

    assert answers[1] == "MULTIPLE"
    assert set(debug[1]["multiple_options"]) == {
        "A",
        "D",
    }


def test_jee_blank_initial_decision_recovers_two_marks():
    gray = np.full(
        (180, 260),
        225,
        dtype=np.uint8,
    )

    centers = {
        "A": (55, 90),
        "B": (105, 90),
        "C": (155, 90),
        "D": (205, 90),
    }

    for center in centers.values():
        cv2.circle(
            gray,
            center,
            10,
            80,
            2,
            lineType=cv2.LINE_AA,
        )

    cv2.circle(
        gray,
        centers["B"],
        7,
        35,
        -1,
        lineType=cv2.LINE_AA,
    )

    cv2.circle(
        gray,
        centers["D"],
        7,
        75,
        -1,
        lineType=cv2.LINE_AA,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    options = {}

    for option, center in centers.items():
        options[option] = {
            "crop_center": [
                center[0],
                center[1],
            ],
            "ml_filled_probability": (
                0.94
                if option == "B"
                else (
                    0.60
                    if option == "D"
                    else 0.03
                )
            ),
            "ml_blank_probability": (
                0.03
                if option in ("B", "D")
                else 0.93
            ),
            "ml_ambiguous_probability": 0.03,
        }

    answer, decision = rescue_jee_multiple_from_ml_debug(
        gray=gray,
        ml_answer=None,
        ml_decision={
            "answer": None,
            "status": "blank",
            "multiple_options": [],
            "options": options,
        },
    )

    assert answer == "MULTIPLE"
    assert set(decision["multiple_options"]) == {
        "B",
        "D",
    }
