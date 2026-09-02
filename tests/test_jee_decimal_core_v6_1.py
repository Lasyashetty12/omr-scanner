from pathlib import Path
import json
import cv2

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


def _fill(image, x, y):
    cv2.circle(
        image,
        (
            int(round(x)),
            int(round(y)),
        ),
        7,
        (35, 35, 35),
        -1,
        lineType=cv2.LINE_AA,
    )


def test_decimal_core_v6_1_reads_q50_q74_q75_exactly():
    from jee_reader import (
        scan_jee_numerical_sections_robust,
    )

    template, image = _load()
    layout = template[
        "numerical_layout"
    ]

    wanted = {
        50: [4],
        74: [2, 5],
        75: [3],
    }

    for section in template[
        "numerical_sections"
    ]:
        start = int(
            section[
                "start_question"
            ]
        )

        for index, base_x in enumerate(
            section[
                "question_x_positions"
            ]
        ):
            question = (
                start + index
            )

            if question not in wanted:
                continue

            for after_column in wanted[
                question
            ]:
                decimal_index = (
                    layout[
                        "decimal_after_columns"
                    ].index(
                        after_column
                    )
                )

                _fill(
                    image,
                    int(base_x)
                    + int(
                        layout[
                            "decimal_offsets"
                        ][
                            decimal_index
                        ]
                    ),
                    int(
                        section[
                            "decimal_y"
                        ]
                    ),
                )

    detected, _debug = (
        scan_jee_numerical_sections_robust(
            image,
            template,
        )
    )

    for question, expected in wanted.items():
        actual = sorted(
            int(
                detail[
                    "after_column"
                ]
            )
            for detail
            in detected[
                question
            ][
                "decimal_points"
            ]
            if bool(
                detail.get(
                    "filled",
                    False,
                )
            )
        )

        assert actual == expected


def test_blank_decimal_row_stays_blank():
    from jee_reader import (
        scan_jee_numerical_sections_robust,
    )

    template, image = _load()

    detected, _debug = (
        scan_jee_numerical_sections_robust(
            image,
            template,
        )
    )

    for question in (
        50,
        74,
        75,
    ):
        actual = [
            detail
            for detail
            in detected[
                question
            ][
                "decimal_points"
            ]
            if bool(
                detail.get(
                    "filled",
                    False,
                )
            )
        ]

        assert actual == []


def test_decimal_core_v6_1_contract():
    source = (
        ROOT
        / "jee_reader.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "decimal_core_fill_v6_1"
        in source
    )

    assert (
        "jee_numeric_decimal_core_threshold"
        in source
    )
