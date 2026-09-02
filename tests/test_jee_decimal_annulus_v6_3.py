from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    template = json.loads(
        (
            ROOT
            / "templates"
            / "jee.json"
        ).read_text(encoding="utf-8")
    )

    image = cv2.imread(
        str(
            ROOT
            / "references"
            / template["reference_image"]
        )
    )

    assert image is not None

    return template, image


def _fill(image, x, y):
    cv2.circle(
        image,
        (
            int(round(x)),
            int(round(y)),
        ),
        7,
        (25, 25, 25),
        -1,
        lineType=cv2.LINE_AA,
    )


def _decimal_positions(
    template,
    question_number,
):
    layout = template[
        "numerical_layout"
    ]

    for section in template[
        "numerical_sections"
    ]:
        start = int(
            section[
                "start_question"
            ]
        )

        xs = section[
            "question_x_positions"
        ]

        if (
            start
            <= question_number
            < start + len(xs)
        ):
            base_x = int(
                xs[
                    question_number
                    - start
                ]
            )

            return {
                int(after_column): (
                    base_x + int(offset),
                    int(
                        section[
                            "decimal_y"
                        ]
                    ),
                )
                for after_column, offset
                in zip(
                    layout[
                        "decimal_after_columns"
                    ],
                    layout[
                        "decimal_offsets"
                    ],
                )
            }

    raise AssertionError(
        "question not found"
    )


def _shift(image, dx=3, dy=-2):
    matrix = np.float32(
        [
            [1, 0, dx],
            [0, 1, dy],
        ]
    )

    return cv2.warpAffine(
        image,
        matrix,
        (
            image.shape[1],
            image.shape[0],
        ),
        borderValue=255,
    )


def test_v6_3_shifted_blank_decimal_rows_have_no_false_fills():
    from jee_reader import (
        scan_jee_numerical_sections_robust,
    )

    template, image = _load()

    image = _shift(
        image,
        dx=3,
        dy=-2,
    )

    detected, _debug = (
        scan_jee_numerical_sections_robust(
            image,
            template,
        )
    )

    for question in (
        71,
        72,
        74,
    ):
        filled = [
            item[
                "after_column"
            ]
            for item
            in detected[
                question
            ][
                "decimal_points"
            ]
            if item.get(
                "filled"
            )
        ]

        assert filled == []


def test_v6_3_q71_first_and_q72_third_only():
    from jee_reader import (
        scan_jee_numerical_sections_robust,
    )

    template, image = _load()

    expected = {
        71: [1],
        72: [3],
    }

    for question, wanted in expected.items():
        positions = _decimal_positions(
            template,
            question,
        )

        for after in wanted:
            x, y = positions[after]
            _fill(
                image,
                x,
                y,
            )

    image = _shift(
        image,
        dx=3,
        dy=-2,
    )

    detected, _debug = (
        scan_jee_numerical_sections_robust(
            image,
            template,
        )
    )

    for question, wanted in expected.items():
        filled = sorted(
            int(
                item[
                    "after_column"
                ]
            )
            for item
            in detected[
                question
            ][
                "decimal_points"
            ]
            if item.get(
                "filled"
            )
        )

        assert filled == wanted


def test_v6_3_q74_multiple_fills_are_preserved():
    from jee_reader import (
        scan_jee_numerical_sections_robust,
    )

    template, image = _load()

    positions = _decimal_positions(
        template,
        74,
    )

    for after in (
        2,
        5,
    ):
        x, y = positions[after]

        _fill(
            image,
            x,
            y,
        )

    image = _shift(
        image,
        dx=3,
        dy=-2,
    )

    detected, _debug = (
        scan_jee_numerical_sections_robust(
            image,
            template,
        )
    )

    record = detected[74]

    filled = sorted(
        int(
            item[
                "after_column"
            ]
        )
        for item
        in record[
            "decimal_points"
        ]
        if item.get(
            "filled"
        )
    )

    assert filled == [
        2,
        5,
    ]

    assert (
        record[
            "decimal_status"
        ]
        == "MULTIPLE"
    )


def test_v6_3_contract():
    reader = (
        ROOT
        / "jee_reader.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "decimal_annulus_fill_v6_3"
        in reader
    )

    assert (
        "individual_hough_center_v6_3"
        in reader
    )

    assert (
        "def _decimal_annulus_fill_ratio("
        in reader
    )
