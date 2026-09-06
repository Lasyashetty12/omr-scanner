from pathlib import Path

import cv2
import numpy as np

from ml_omr.row_ring_alignment import (
    refine_fitted_grid_to_printed_rings,
)


ROOT = Path(__file__).resolve().parents[1]


def _kcet_template():
    return {
        "exam_name": "KCET",
        "questions_per_column": 60,
        "bubble_radius": 11,
        "grid_max_final_dy_from_input": 6,
    }


def _base_row():
    return {
        1: {
            "A": (80.0, 90.0),
            "B": (130.0, 90.0),
            "C": (180.0, 90.0),
            "D": (230.0, 90.0),
        },
    }


def test_shared_row_refinement_locks_to_real_printed_rings():
    image = np.full(
        (180, 320),
        245,
        dtype=np.uint8,
    )

    fitted = _base_row()
    calibrated = _base_row()

    expected_dx = 2
    expected_dy = -2

    for center_x, center_y in fitted[1].values():
        cv2.circle(
            image,
            (
                int(center_x) + expected_dx,
                int(center_y) + expected_dy,
            ),
            10,
            65,
            2,
            lineType=cv2.LINE_AA,
        )

    # One option is filled. It must not be allowed to move independently.
    cv2.circle(
        image,
        (
            180 + expected_dx,
            90 + expected_dy,
        ),
        7,
        30,
        -1,
        lineType=cv2.LINE_AA,
    )

    refined, debug = (
        refine_fitted_grid_to_printed_rings(
            image,
            fitted,
            calibrated,
            _kcet_template(),
        )
    )

    shifts = {
        (
            int(
                round(
                    refined[1][option][0]
                    - fitted[1][option][0]
                )
            ),
            int(
                round(
                    refined[1][option][1]
                    - fitted[1][option][1]
                )
            ),
        )
        for option
        in ("A", "B", "C", "D")
    }

    assert len(shifts) == 1

    shift_x, shift_y = next(iter(shifts))

    assert abs(shift_x - expected_dx) <= 1
    assert abs(shift_y - expected_dy) <= 1
    assert debug[0]["applied_rows"] == 1
    assert (
        debug[0]["rows"][0]["improved_options"]
        >= 3
    )


def test_shared_row_refinement_cannot_make_large_row_jump():
    image = np.full(
        (180, 320),
        245,
        dtype=np.uint8,
    )

    fitted = _base_row()
    calibrated = _base_row()

    # Tempt the detector with a complete row 10 px away. The v10.22 layer
    # is only allowed to search locally and cannot chase the neighboring row.
    for center_x, center_y in fitted[1].values():
        cv2.circle(
            image,
            (
                int(center_x),
                int(center_y) + 10,
            ),
            10,
            65,
            2,
            lineType=cv2.LINE_AA,
        )

    refined, _debug = (
        refine_fitted_grid_to_printed_rings(
            image,
            fitted,
            calibrated,
            _kcet_template(),
        )
    )

    for option in ("A", "B", "C", "D"):
        dx = (
            refined[1][option][0]
            - fitted[1][option][0]
        )
        dy = (
            refined[1][option][1]
            - fitted[1][option][1]
        )

        assert abs(dx) <= 3
        assert abs(dy) <= 3


def test_scanner_uses_shared_ring_alignment_and_full_overlay_radius():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "refine_fitted_grid_to_printed_rings("
        in source
    )

    assert (
        'base_radius = max(3.0, float(template.get("bubble_radius", 11)))'
        in source
    )

    assert (
        'float(template.get("bubble_radius", 11)) - 6.0'
        not in source
    )


def test_kcet_keeps_safe_vertical_grid_limit():
    import json

    template = json.loads(
        (
            ROOT
            / "templates"
            / "kcet.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert template[
        "grid_max_direct_pin_dy"
    ] == 6

    assert template[
        "grid_max_final_dy_from_input"
    ] == 6
