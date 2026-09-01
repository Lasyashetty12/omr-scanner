from pathlib import Path
import json

import cv2

from jee_precise_reader import scan_jee_answers_precise


ROOT = Path(__file__).resolve().parents[1]


def _fill(image, x, y, radius=7):
    cv2.circle(
        image,
        (int(x), int(y)),
        int(radius),
        (0, 0, 0),
        -1,
        lineType=cv2.LINE_AA,
    )


def _mark_mcq(image, template, question_number, option):
    for section in template["mcq_sections"]:
        start = int(section["start_question"])
        total = int(section["total_questions"])

        if start <= question_number < start + total:
            row = question_number - start

            _fill(
                image,
                section["option_x"][option],
                section["question_y_positions"][row],
            )
            return

    raise AssertionError(
        f"No MCQ section for Q{question_number}"
    )


def _mark_numeric(image, template, question_number, answer):
    layout = template["numerical_layout"]

    for section in template["numerical_sections"]:
        start = int(section["start_question"])
        count = len(section["question_x_positions"])

        if not (
            start
            <= question_number
            < start + count
        ):
            continue

        index = question_number - start
        base_x = int(
            section["question_x_positions"][index]
        )

        text = str(answer)
        negative = text.startswith("-")

        if negative:
            text = text[1:]

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

        if "." in text:
            integer, fraction = text.split(".", 1)
            digits = integer + fraction
            decimal_after = len(integer)
        else:
            digits = text
            decimal_after = None

        for column_index, digit in enumerate(digits):
            _fill(
                image,
                base_x
                + int(
                    layout[
                        "digit_offsets"
                    ][column_index]
                ),
                int(
                    section[
                        "digit_y_positions"
                    ][int(digit)]
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
                    ][decimal_index]
                ),
                int(section["decimal_y"]),
            )

        return

    raise AssertionError(
        f"No numerical section for Q{question_number}"
    )


def test_precise_jee_reader_reads_all_75_key_answers():
    image = cv2.imread(
        str(
            ROOT
            / "references"
            / "jee_generated.png"
        )
    )

    assert image is not None

    template = json.loads(
        (
            ROOT
            / "templates"
            / "jee.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    key = json.loads(
        (
            ROOT
            / "answer_keys"
            / "jee"
            / "P.json"
        ).read_text(
            encoding="utf-8"
        )
    )["answers"]

    for question, answer in key["mcq"].items():
        _mark_mcq(
            image,
            template,
            int(question),
            answer,
        )

    for question, answer in key["numerical"].items():
        _mark_numeric(
            image,
            template,
            int(question),
            answer,
        )

    detected = scan_jee_answers_precise(
        image,
        template,
    )

    mcq_errors = {
        question: (
            detected["mcq"][
                int(question)
            ]["answer"],
            expected,
        )
        for question, expected
        in key["mcq"].items()
        if (
            detected["mcq"][
                int(question)
            ]["answer"]
            != expected
        )
    }

    numerical_errors = {
        question: (
            detected["numerical"][
                int(question)
            ]["answer"],
            expected,
        )
        for question, expected
        in key["numerical"].items()
        if (
            detected["numerical"][
                int(question)
            ]["answer"]
            != expected
        )
    }

    assert mcq_errors == {}
    assert numerical_errors == {}
