import cv2
import numpy as np

from ml_omr.hybrid_reader import (
    _bubble_metrics,
    _postprocess_known_failure_classes,
    crop_bubble,
)


def _make_option_data(gray, centers, filled):
    data = {}

    for option, center in centers.items():
        crop = crop_bubble(
            gray,
            center[0],
            center[1],
            16,
        )
        metrics = _bubble_metrics(
            crop
        )

        h, w = crop.shape[:2]
        cy = h // 2
        cx = w // 2
        r = 3

        tiny = crop[
            cy - r:
            cy + r + 1,
            cx - r:
            cx + r + 1,
        ]

        yy, xx = np.ogrid[
            -r:r + 1,
            -r:r + 1,
        ]
        mask = (
            xx * xx
            + yy * yy
            <= r * r
        )

        micro = float(
            255.0
            - np.mean(
                tiny[
                    mask
                ]
            )
        )

        data[option] = {
            "crop_center": [
                float(center[0]),
                float(center[1]),
            ],
            "metrics": metrics,
            "micro_core_darkness": micro,
            "ml_filled_probability":
                0.95
                if option in filled
                else 0.02,
            "ml_blank_probability":
                0.02
                if option in filled
                else 0.95,
            "ml_ambiguous_probability": 0.0,
        }

    return data


def _synthetic_row(second_fill=True):
    gray = np.full(
        (140, 210),
        235,
        dtype=np.uint8,
    )

    centers = {
        "A": (45, 70),
        "B": (85, 70),
        "C": (125, 70),
        "D": (165, 70),
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

    # Strong first mark.
    cv2.circle(
        gray,
        centers["A"],
        8,
        30,
        -1,
        lineType=cv2.LINE_AA,
    )

    # Weaker second mark, deliberately unlike the first.
    if second_fill:
        cv2.circle(
            gray,
            centers["C"],
            6,
            40,
            -1,
            lineType=cv2.LINE_AA,
        )

    return gray, centers


def test_jee_answered_row_can_be_rescued_to_multiple():
    gray, centers = _synthetic_row(
        second_fill=True
    )

    option_data = _make_option_data(
        gray,
        centers,
        {"A", "C"},
    )

    # Mimic the real failure: the first pass kept only C.
    decision = {
        "answer": "C",
        "status": "answered",
        "best_option": "C",
        "multiple_options": [],
    }

    result = _postprocess_known_failure_classes(
        16,
        option_data,
        decision,
        gray,
        questions_per_column=1000,
        crop_radius=10,
    )

    assert result[
        "status"
    ] == "multiple"

    assert set(
        result[
            "multiple_options"
        ]
    ) == {
        "A",
        "C",
    }

    assert result[
        "jee_hidden_multiple_rescue"
    ] is True


def test_jee_single_mark_stays_single():
    gray, centers = _synthetic_row(
        second_fill=False
    )

    option_data = _make_option_data(
        gray,
        centers,
        {"A"},
    )

    decision = {
        "answer": "A",
        "status": "answered",
        "best_option": "A",
        "multiple_options": [],
    }

    result = _postprocess_known_failure_classes(
        60,
        option_data,
        decision,
        gray,
        questions_per_column=1000,
        crop_radius=10,
    )

    assert result[
        "status"
    ] == "answered"

    assert result[
        "answer"
    ] == "A"
