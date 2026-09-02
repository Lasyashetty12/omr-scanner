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


def _fill(
    image,
    x,
    y,
    *,
    value=25,
    dx=0,
    dy=0,
):
    cv2.circle(
        image,
        (
            int(round(x + dx)),
            int(round(y + dy)),
        ),
        7,
        (
            int(value),
            int(value),
            int(value),
        ),
        -1,
        lineType=cv2.LINE_AA,
    )


def test_v4_blank_reference_stays_blank_after_mobile_tone_change():
    from jee_reader import (
        scan_jee_numerical_sections_robust,
    )

    template, image = _load()

    mobile = cv2.GaussianBlur(
        image,
        (3, 3),
        0,
    )

    mobile = np.clip(
        mobile.astype(np.float32) * 0.94 + 5.0,
        0,
        255,
    ).astype(np.uint8)

    detected, _debug = (
        scan_jee_numerical_sections_robust(
            mobile,
            template,
        )
    )

    wrong = {
        q: record["answer"]
        for q, record
        in detected.items()
        if record["answer"] != "BLANK"
    }

    assert wrong == {}


def test_v4_reads_faint_shifted_numeric_bubbles():
    from jee_reader import (
        scan_jee_numerical_sections_robust,
    )

    template, image = _load()

    expected = json.loads(
        (
            ROOT
            / "answer_keys"
            / "jee"
            / "P.json"
        ).read_text(encoding="utf-8")
    )["answers"]["numerical"]

    layout = template["numerical_layout"]

    for section in template[
        "numerical_sections"
    ]:
        start = int(
            section["start_question"]
        )

        for index, base_x_value in enumerate(
            section["question_x_positions"]
        ):
            q = start + index
            answer = str(
                expected[str(q)]
            )

            base_x = int(
                base_x_value
            )

            value = answer

            dx = (
                index % 3
            ) - 1

            dy = (
                (index + 1) % 3
            ) - 1

            if value.startswith("-"):
                value = value[1:]

                _fill(
                    image,
                    base_x
                    + int(
                        layout.get(
                            "sign_offset",
                            0,
                        )
                    ),
                    int(section["sign_y"]),
                    value=55,
                    dx=dx,
                    dy=dy,
                )

            if "." in value:
                integer, fraction = (
                    value.split(".", 1)
                )
                digits = integer + fraction
                decimal_after = len(
                    integer
                )
            else:
                digits = value
                decimal_after = None

            for column_index, digit in enumerate(
                digits
            ):
                _fill(
                    image,
                    base_x
                    + int(
                        layout["digit_offsets"][
                            column_index
                        ]
                    ),
                    int(
                        section[
                            "digit_y_positions"
                        ][
                            int(digit)
                        ]
                    ),
                    value=55,
                    dx=dx,
                    dy=dy,
                )

            if decimal_after is not None:
                decimal_index = (
                    layout[
                        "decimal_after_columns"
                    ].index(
                        decimal_after
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
                    int(section["decimal_y"]),
                    value=55,
                    dx=dx,
                    dy=dy,
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

    for question, expected_value in expected.items():
        actual = (
            detected[
                int(question)
            ]["answer"]
        )

        if actual != expected_value:
            errors[question] = (
                actual,
                expected_value,
            )

    assert errors == {}


def test_v4_reader_contract():
    source = (
        ROOT
        / "jee_reader.py"
    ).read_text(encoding="utf-8")

    assert (
        "blank_reference_adaptive_v4"
        in source
    )

    assert (
        "rescued_from_uncertain"
        in source
    )

    assert (
        "jee_adaptive_reference_reader_v4"
        in source
    )
