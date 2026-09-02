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
        ).read_text(
            encoding="utf-8"
        )
    )

    image = cv2.imread(
        str(
            ROOT
            / "references"
            / template[
                "reference_image"
            ]
        )
    )

    assert image is not None

    return template, image


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
                    base_x
                    + int(offset),
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


def _fill(
    image,
    x,
    y,
):
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


def _mobile_shadow(
    image,
):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    ).astype(
        np.float32
    )

    h, w = gray.shape[:2]

    gradient = np.linspace(
        0.58,
        0.72,
        w,
        dtype=np.float32,
    )[None, :]

    gray = (
        gray
        * gradient
        + 8.0
    )

    gray = np.clip(
        gray,
        0,
        255,
    ).astype(
        np.uint8
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    return cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )


def _filled_after_columns(
    detected,
    question,
):
    return sorted(
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
        if bool(
            item.get(
                "filled",
                False,
            )
        )
    )


def test_v6_4_mobile_shadow_reads_q71_and_q72_exactly():
    from jee_reader import (
        scan_jee_numerical_sections_robust,
    )

    template, image = _load()

    expected = {
        71: [1],
        72: [3],
    }

    for question, wanted in expected.items():
        positions = (
            _decimal_positions(
                template,
                question,
            )
        )

        for after in wanted:
            x, y = positions[
                after
            ]

            _fill(
                image,
                x,
                y,
            )

    image = _mobile_shadow(
        image
    )

    detected, _debug = (
        scan_jee_numerical_sections_robust(
            image,
            template,
        )
    )

    assert (
        _filled_after_columns(
            detected,
            71,
        )
        == [1]
    )

    assert (
        _filled_after_columns(
            detected,
            72,
        )
        == [3]
    )


def test_v6_4_mobile_shadow_preserves_other_known_decimals():
    from jee_reader import (
        scan_jee_numerical_sections_robust,
    )

    template, image = _load()

    expected = {
        50: [4],
        74: [2, 5],
        75: [3],
    }

    for question, wanted in expected.items():
        positions = (
            _decimal_positions(
                template,
                question,
            )
        )

        for after in wanted:
            x, y = positions[
                after
            ]

            _fill(
                image,
                x,
                y,
            )

    image = _mobile_shadow(
        image
    )

    detected, _debug = (
        scan_jee_numerical_sections_robust(
            image,
            template,
        )
    )

    for question, wanted in expected.items():
        assert (
            _filled_after_columns(
                detected,
                question,
            )
            == wanted
        )


def test_v6_4_dark_blank_sheet_has_zero_decimal_fills():
    from jee_reader import (
        scan_jee_numerical_sections_robust,
    )

    template, image = _load()

    image = _mobile_shadow(
        image
    )

    detected, _debug = (
        scan_jee_numerical_sections_robust(
            image,
            template,
        )
    )

    for question in (
        21,
        22,
        23,
        24,
        25,
        46,
        47,
        48,
        49,
        50,
        71,
        72,
        73,
        74,
        75,
    ):
        assert (
            _filled_after_columns(
                detected,
                question,
            )
            == []
        )


def test_v6_4_contract():
    source = (
        ROOT
        / "jee_reader.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "def _decimal_local_contrast_metrics("
        in source
    )

    assert (
        "decimal_local_contrast_v6_4"
        in source
    )

    assert (
        "local_background_relative_v6_4"
        in source
    )
