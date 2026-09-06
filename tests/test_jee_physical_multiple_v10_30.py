import cv2
import numpy as np

from ml_omr.final_guard_v10_29 import (
    resolve_strict_jee_secondary_multiple,
)


def _row(
    filled_options,
    *,
    faint_option=None,
):
    gray = np.full(
        (180, 270),
        225,
        dtype=np.uint8,
    )

    centers = {
        "A": (55, 90),
        "B": (105, 90),
        "C": (155, 90),
        "D": (205, 90),
    }

    # Printed empty ring + a tiny center character, approximating the real JEE
    # bubble artwork. This must NOT be mistaken for a fill.
    for option, center in centers.items():
        cv2.circle(
            gray,
            center,
            10,
            82,
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            gray,
            (center[0], center[1] - 2),
            (center[0], center[1] + 2),
            105,
            1,
            lineType=cv2.LINE_AA,
        )

    for option in filled_options:
        value = (
            78
            if option == faint_option
            else 32
        )

        cv2.circle(
            gray,
            centers[option],
            7,
            value,
            -1,
            lineType=cv2.LINE_AA,
        )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    option_data = {}

    for option, center in centers.items():
        is_fill = option in filled_options

        # Intentionally give a weak ML score to the faint real bubble to
        # reproduce the production miss.
        if option == faint_option:
            ml_filled = 0.28
            ml_blank = 0.62
        elif is_fill:
            ml_filled = 0.91
            ml_blank = 0.05
        else:
            ml_filled = 0.03
            ml_blank = 0.94

        option_data[option] = {
            "crop_center": [
                center[0],
                center[1],
            ],
            "ml_filled_probability":
                ml_filled,
            "ml_blank_probability":
                ml_blank,
            "ml_ambiguous_probability":
                max(
                    0.0,
                    1.0
                    - ml_filled
                    - ml_blank,
                ),
            "metrics": {
                "disk_dark_ratio":
                    (
                        0.86
                        if is_fill
                        else 0.31
                    ),
                "center_darkness":
                    (
                        118.0
                        if is_fill
                        else 39.0
                    ),
                "core_dark_ratio":
                    (
                        0.91
                        if is_fill
                        else 0.22
                    ),
            },
        }

    return gray, option_data


def test_stable_single_gets_one_dark_secondary_fill():
    gray, option_data = _row(
        {"B", "D"},
        faint_option="B",
    )

    answer, decision = (
        resolve_strict_jee_secondary_multiple(
            stable_answer="D",
            ml_answer="D",
            ml_decision={
                "answer": "D",
                "status": "answered",
                "options": option_data,
            },
            gray=gray,
        )
    )

    assert answer == "MULTIPLE"
    assert set(
        decision[
            "multiple_options"
        ]
    ) == {"B", "D"}


def test_blank_can_recover_only_two_strong_physical_fills():
    gray, option_data = _row(
        {"A", "C"},
    )

    answer, decision = (
        resolve_strict_jee_secondary_multiple(
            stable_answer="BLANK",
            ml_answer=None,
            ml_decision={
                "answer": None,
                "status": "blank",
                "options": option_data,
            },
            gray=gray,
        )
    )

    assert answer == "MULTIPLE"
    assert set(
        decision[
            "multiple_options"
        ]
    ) == {"A", "C"}


def test_one_real_fill_stays_single():
    gray, option_data = _row(
        {"C"},
    )

    answer, decision = (
        resolve_strict_jee_secondary_multiple(
            stable_answer="C",
            ml_answer="C",
            ml_decision={
                "answer": "C",
                "status": "answered",
                "options": option_data,
            },
            gray=gray,
        )
    )

    assert answer == "C"
    assert decision["status"] == "answered"


def test_empty_printed_rings_never_create_multiple():
    gray, option_data = _row(
        set(),
    )

    answer, decision = (
        resolve_strict_jee_secondary_multiple(
            stable_answer="BLANK",
            ml_answer=None,
            ml_decision={
                "answer": None,
                "status": "blank",
                "options": option_data,
            },
            gray=gray,
        )
    )

    assert answer is None
    assert decision["status"] == "blank"


def test_three_suspicious_options_are_rejected_not_multiple():
    gray, option_data = _row(
        {"A", "B", "D"},
    )

    answer, decision = (
        resolve_strict_jee_secondary_multiple(
            stable_answer="A",
            ml_answer="A",
            ml_decision={
                "answer": "A",
                "status": "answered",
                "options": option_data,
            },
            gray=gray,
        )
    )

    assert answer == "A"
    assert decision["status"] == "answered"
