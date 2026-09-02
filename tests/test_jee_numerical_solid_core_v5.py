from pathlib import Path
import json
import cv2

ROOT = Path(__file__).resolve().parents[1]

DIGIT_PATTERN = {
    21: [2, 2, 1, 4, None, 2, 4],
    22: [4, 0, 2, 4, 1, 3, 5],
    23: [0, 1, 3, None, 4, None, 9],
    24: [2, None, 3, 5, 3, 1, 7],
    25: [2, 3, 5, 2, 1, None, 2],

    46: [3, 2, None, 2, 4, 6, 1],
    47: [2, None, 4, 4, 4, 0, 1],
    48: [3, None, 2, 2, 4, 6, 3],
    49: [4, 3, 1, 0, 2, None, 3],
    50: [4, 2, 2, 1, None, 1, 9],

    71: [0, None, 2, 5, 6, 2, None],
    72: [4, 0, 1, None, 2, 4, 6],
    73: [4, None, 2, 3, 4, 9, None],
    74: [1, 2, 2, 5, 2, 3, 9],
    75: [2, 1, 1, None, 3, 5, 8],
}

DECIMAL_PATTERN = {
    21: [4],
    22: [4],
    23: [3],
    24: [1],
    25: [5],

    46: [2],
    47: [1],
    48: [1],
    49: [5],
    50: [4],

    71: [1],
    72: [3],
    73: [1],
    74: [2, 5],
    75: [3],
}

SIGN_PATTERN = {
    22,
    25,
    50,
    73,
    74,
}


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
        (
            35,
            35,
            35,
        ),
        -1,
        lineType=cv2.LINE_AA,
    )


def test_v5_matches_stress_pattern_without_false_bubbles():
    from jee_reader import (
        scan_jee_numerical_sections_robust,
    )

    template, image = _load()

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

        for index, base_x_value in enumerate(
            section[
                "question_x_positions"
            ]
        ):
            question = (
                start
                + index
            )

            base_x = int(
                base_x_value
            )

            for column_index, digit in enumerate(
                DIGIT_PATTERN[
                    question
                ]
            ):
                if digit is None:
                    continue

                _fill(
                    image,
                    base_x
                    + int(
                        layout[
                            "digit_offsets"
                        ][
                            column_index
                        ]
                    ),
                    int(
                        section[
                            "digit_y_positions"
                        ][
                            int(
                                digit
                            )
                        ]
                    ),
                )

            for after_column in DECIMAL_PATTERN[
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
                    base_x
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

            if question in SIGN_PATTERN:
                _fill(
                    image,
                    base_x
                    + int(
                        layout.get(
                            "sign_offset",
                            0,
                        )
                    ),
                    int(
                        section[
                            "sign_y"
                        ]
                    ),
                )

    image = cv2.GaussianBlur(
        image,
        (3, 3),
        0,
    )

    detected, _debug = (
        scan_jee_numerical_sections_robust(
            image,
            template,
        )
    )

    errors = {}

    for question, expected_columns in DIGIT_PATTERN.items():
        record = detected[
            question
        ]

        actual_columns = []

        for detail in record[
            "columns"
        ]:
            candidates = (
                detail.get(
                    "filled_candidates",
                    [],
                )
            )

            if len(candidates) == 1:
                actual_columns.append(
                    int(
                        candidates[0][
                            "value"
                        ]
                    )
                )
            elif len(candidates) == 0:
                actual_columns.append(
                    None
                )
            else:
                actual_columns.append(
                    "MULTIPLE"
                )

        actual_decimals = sorted(
            int(
                detail[
                    "after_column"
                ]
            )
            for detail
            in record[
                "decimal_points"
            ]
            if bool(
                detail.get(
                    "filled",
                    False,
                )
            )
        )

        actual_sign = bool(
            (
                record.get(
                    "sign"
                )
                or {}
            ).get(
                "filled",
                False,
            )
        )

        expected_sign = (
            question
            in SIGN_PATTERN
        )

        if (
            actual_columns
            != expected_columns
            or actual_decimals
            != DECIMAL_PATTERN[
                question
            ]
            or actual_sign
            != expected_sign
        ):
            errors[
                question
            ] = {
                "columns":
                    (
                        actual_columns,
                        expected_columns,
                    ),

                "decimals":
                    (
                        actual_decimals,
                        DECIMAL_PATTERN[
                            question
                        ],
                    ),

                "sign":
                    (
                        actual_sign,
                        expected_sign,
                    ),
            }

    assert errors == {}


def test_v5_marks_two_decimals_as_ambiguous():
    from jee_reader import (
        scan_jee_numerical_sections_robust,
    )

    template, image = _load()

    section = template[
        "numerical_sections"
    ][2]

    layout = template[
        "numerical_layout"
    ]

    base_x = int(
        section[
            "question_x_positions"
        ][3]
    )

    for after_column in (
        2,
        5,
    ):
        decimal_index = (
            layout[
                "decimal_after_columns"
            ].index(
                after_column
            )
        )

        _fill(
            image,
            base_x
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

    record = detected[74]

    assert (
        record[
            "decimal_status"
        ]
        == "MULTIPLE"
    )

    assert (
        record[
            "answer"
        ]
        == "UNCERTAIN"
    )


def test_v5_overlay_is_per_bubble():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "filled_candidates"
        in source
    )

    assert (
        "filled_decimal_points"
        in source
    )

    assert (
        "Numerical bubble confidence is per bubble"
        in source
    )
