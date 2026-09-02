from pathlib import Path
import json
import cv2

ROOT = Path(__file__).resolve().parents[1]


def _fill(image, x, y):
    cv2.circle(
        image,
        (int(x), int(y)),
        7,
        (0, 0, 0),
        -1,
        lineType=cv2.LINE_AA,
    )


def test_precise_numeric_reader_reads_all_15_jee_values():
    from jee_precise_reader import (
        scan_jee_numerical_precise,
    )

    template = json.loads(
        (
            ROOT
            / "templates"
            / "jee.json"
        ).read_text(encoding="utf-8")
    )

    expected = json.loads(
        (
            ROOT
            / "answer_keys"
            / "jee"
            / "P.json"
        ).read_text(encoding="utf-8")
    )["answers"]["numerical"]

    image = cv2.imread(
        str(
            ROOT
            / "references"
            / "jee_generated.png"
        )
    )

    assert image is not None

    layout = template["numerical_layout"]

    for section in template["numerical_sections"]:
        start = int(section["start_question"])

        for index, base_x_value in enumerate(
            section["question_x_positions"]
        ):
            q = start + index
            answer = str(expected[str(q)])
            base_x = int(base_x_value)
            value = answer

            if value.startswith("-"):
                value = value[1:]
                _fill(
                    image,
                    base_x + int(layout.get("sign_offset", 0)),
                    int(section["sign_y"]),
                )

            if "." in value:
                integer, fraction = value.split(".", 1)
                digits = integer + fraction
                decimal_after = len(integer)
            else:
                digits = value
                decimal_after = None

            for column_index, digit in enumerate(digits):
                _fill(
                    image,
                    base_x
                    + int(
                        layout["digit_offsets"][column_index]
                    ),
                    int(
                        section["digit_y_positions"][int(digit)]
                    ),
                )

            if decimal_after is not None:
                decimal_index = (
                    layout["decimal_after_columns"]
                    .index(decimal_after)
                )

                _fill(
                    image,
                    base_x
                    + int(
                        layout["decimal_offsets"][decimal_index]
                    ),
                    int(section["decimal_y"]),
                )

    detected, calibration = scan_jee_numerical_precise(
        image,
        template,
    )

    errors = {}

    for question, expected_value in expected.items():
        actual = detected[int(question)]["answer"]
        if actual != expected_value:
            errors[question] = (
                actual,
                expected_value,
            )

    assert errors == {}
    assert len(calibration) == 15


def test_production_jee_keeps_mcq_baseline_and_overrides_numeric_only():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(encoding="utf-8")

    process_start = source.index(
        "def process_omr("
    )

    start = source.index(
        '    elif exam_name == "JEE":',
        process_start,
    )

    end = source.index(
        "\n    else:",
        start,
    )

    branch = source[start:end]

    assert "scan_jee_answers(" in branch
    assert "scan_jee_numerical_sections_robust(" in branch
    assert 'answers["numerical"]' in branch

    assert "scan_jee_numerical_precise(" not in branch
    assert "scan_jee_answers_precise(" not in branch
    assert "scan_jee_mcq_precise(" not in branch
