import cv2
import numpy as np

from ml_omr.cv_lattice_alignment import (
    refine_fitted_grid_to_printed_rings,
)


def _synthetic_kcet():
    options = ["A", "B", "C", "D"]
    seed_x = {
        "A": 574,
        "B": 616,
        "C": 658,
        "D": 699,
    }

    seed = {}
    actual = {}

    image = np.full(
        (2263, 1600),
        245,
        dtype=np.uint8,
    )

    for row in range(60):
        question = row + 1
        seed_y = 326 + row * 28
        physical_y = 338 + row * 28 + 2.0 * np.sin(row / 8.0)

        seed[question] = {}
        actual[question] = {}

        for lane, option in enumerate(options):
            physical_x = (
                [581, 623, 665, 704][lane]
                + 1.5 * np.sin(row / 10.0 + lane)
            )

            seed[question][option] = (
                float(seed_x[option]),
                float(seed_y),
            )
            actual[question][option] = (
                float(physical_x),
                float(physical_y),
            )

            cv2.circle(
                image,
                (
                    int(round(physical_x)),
                    int(round(physical_y)),
                ),
                10,
                70,
                2,
                lineType=cv2.LINE_AA,
            )

    template = {
        "exam_name": "KCET",
        "questions_per_column": 60,
        "options": options,
        "columns": [seed_x],
        "bubble_radius": 11,
    }

    fallback = {
        q: {
            option: center
            for option, center in option_map.items()
        }
        for q, option_map in seed.items()
    }

    return image, template, fallback, seed, actual


def test_cv_lattice_is_geometry_authority_not_template_offset():
    image, template, fallback, seed, actual = _synthetic_kcet()

    fitted, debug = refine_fitted_grid_to_printed_rings(
        image,
        fallback,
        seed,
        template,
    )

    assert debug[0]["status"] == "cv_lattice_authority"
    assert debug[0]["detected_row_count"] == 60
    assert debug[0]["direct_cell_coverage"] >= 0.98

    # Physical sheet is deliberately >6 px away from template Y.
    # The result must follow the real ring rather than stay template-clamped.
    assert abs(fitted[1]["A"][1] - seed[1]["A"][1]) > 6.0
    assert abs(fitted[1]["A"][0] - actual[1]["A"][0]) <= 2.0
    assert abs(fitted[1]["A"][1] - actual[1]["A"][1]) <= 2.0

    # One question row must remain horizontal across A/B/C/D.
    y_values = [
        fitted[30][option][1]
        for option in template["options"]
    ]
    assert max(y_values) - min(y_values) <= 0.01


def test_jee_geometry_is_not_changed():
    fitted = {
        1: {
            "A": (100.0, 100.0),
            "B": (120.0, 100.0),
            "C": (140.0, 100.0),
            "D": (160.0, 100.0),
        }
    }

    image = np.full((300, 300), 255, dtype=np.uint8)

    output, debug = refine_fitted_grid_to_printed_rings(
        image,
        fitted,
        fitted,
        {
            "exam_name": "JEE",
            "columns": [],
        },
    )

    assert output == fitted
    assert debug == {}
