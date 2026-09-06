import cv2
import numpy as np

from jee_cv_mcq_reader import (
    scan_jee_mcq_sections_cv_robust,
)


def _template():
    return {
        "bubble_radius": 10,
        "mcq_sections": [
            {
                "subject": "physics",
                "start_question": 1,
                "total_questions": 10,
                "option_x": {
                    "A": 487,
                    "B": 517,
                    "C": 546,
                    "D": 578,
                },
                "question_y_positions": [
                    426,
                    460,
                    496,
                    530,
                    564,
                    598,
                    634,
                    666,
                    704,
                    738,
                ],
            }
        ],
    }


def test_jee_mcq_uses_physical_circle_centres():
    template = _template()

    image = np.full(
        (2263, 1600),
        245,
        dtype=np.uint8,
    )

    options = ["A", "B", "C", "D"]
    seed_x = template[
        "mcq_sections"
    ][0]["option_x"]
    seed_y = template[
        "mcq_sections"
    ][0]["question_y_positions"]

    # Deliberately move the printed lattice beyond the old tiny local
    # refinement window. Also omit three circle contours so interpolation is
    # exercised.
    missing = {
        (2, 1),
        (5, 0),
        (7, 3),
    }

    for row, y in enumerate(seed_y):
        physical_y = (
            float(y)
            + 10.0
            + 1.5
            * np.sin(
                row / 3.0
            )
        )

        for lane, option in enumerate(options):
            if (row, lane) in missing:
                continue

            physical_x = (
                float(seed_x[option])
                - 4.0
                + 1.0
                * np.sin(
                    row / 4.0
                    + lane
                )
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

    records, debug = (
        scan_jee_mcq_sections_cv_robust(
            image,
            template,
        )
    )

    section = debug["sections"]["1"]

    assert (
        section["status"]
        == "cv_physical_circle_authority"
    )
    assert section[
        "covered_row_count"
    ] == 10
    assert section[
        "direct_cell_coverage"
    ] >= 0.90

    # Q1 should follow the real printed grid, not the JSON coordinate.
    q1_a = records[1][
        "option_centres"
    ]["A"]

    assert (
        abs(
            q1_a[1]
            - seed_y[0]
        )
        >= 8
    )

    # Missing cells must still receive a CV-lattice interpolated centre.
    q3_b = records[3][
        "option_centres"
    ]["B"]

    assert len(q3_b) == 2
    assert records[3][
        "centre_sources"
    ]["B"] == (
        "cv_lattice_interpolation"
    )


def test_scanner_switches_jee_mcq_to_cv_reader():
    from pathlib import Path

    root = Path(
        __file__
    ).resolve().parents[1]

    source = (
        root
        / "scanner.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "from jee_cv_mcq_reader import ("
        in source
    )
    assert (
        "scan_jee_mcq_sections_cv_robust "
        "as scan_jee_mcq_sections_robust"
        in source
    )
