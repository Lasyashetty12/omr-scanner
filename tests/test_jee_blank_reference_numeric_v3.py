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
        (0, 0, 0),
        -1,
        lineType=cv2.LINE_AA,
    )


def test_blank_jee_numerical_sheet_has_no_false_digits():
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

    wrong = {
        q: record["answer"]
        for q, record in detected.items()
        if record["answer"] != "BLANK"
    }

    assert wrong == {}


def test_reference_delta_reads_all_15_synthetic_numericals():
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
                )

            if "." in value:
                integer, fraction = (
                    value.split(".", 1)
                )
                digits = integer + fraction
                decimal_after = len(integer)
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


def test_reader_is_reference_delta_v3():
    source = (
        ROOT
        / "jee_reader.py"
    ).read_text(encoding="utf-8")

    assert "_extra_ink_score(" in source
    assert "blank_reference_extra_ink" in source
    assert "jee_blank_reference_reader_v3" in source
