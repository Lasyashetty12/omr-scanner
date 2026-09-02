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
        (int(round(x)), int(round(y))),
        7,
        (35, 35, 35),
        -1,
        lineType=cv2.LINE_AA,
    )


def test_q46_blank_printed_8_is_not_false_fill_with_small_drift():
    from jee_reader import _classify_numerical_column

    template, image = _load()

    section = template["numerical_sections"][1]
    layout = template["numerical_layout"]

    base_x = float(section["question_x_positions"][0])

    column = {
        "x":
            base_x
            + float(layout["digit_offsets"][2])
            - 2.0,
        "y_positions": [
            float(value) - 2.0
            for value in section["digit_y_positions"]
        ],
        "values": list(range(10)),
    }

    record = _classify_numerical_column(
        image,
        column,
        template,
    )

    assert record["filled_candidates"] == []
    assert record["value"] == ""


def test_correct_decimal_rows_q50_q74_q75():
    from jee_reader import scan_jee_numerical_sections_robust

    template, image = _load()
    layout = template["numerical_layout"]

    wanted = {
        50: [4],
        74: [2, 5],
        75: [3],
    }

    for section in template["numerical_sections"]:
        start = int(section["start_question"])

        for index, base_x in enumerate(
            section["question_x_positions"]
        ):
            question = start + index

            if question not in wanted:
                continue

            for after_column in wanted[question]:
                decimal_index = (
                    layout["decimal_after_columns"]
                    .index(after_column)
                )

                _fill(
                    image,
                    int(base_x)
                    + int(
                        layout["decimal_offsets"][
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

    for question, expected in wanted.items():
        actual = sorted(
            int(detail["after_column"])
            for detail
            in detected[question]["decimal_points"]
            if bool(detail.get("filled", False))
        )

        assert actual == expected


def test_uniform_core_v6_contract():
    source = (
        ROOT
        / "jee_reader.py"
    ).read_text(encoding="utf-8")

    assert "uniform_core_v6" in source
    assert "actual_decimal_y" in source

    # v6.1 intentionally replaced decimal p90 classification with
    # direct decimal core-fill scoring.
    assert "decimal_core_fill_v6_1" in source
    assert "jee_numeric_decimal_core_threshold" in source
    assert "jee_numeric_decimal_p90_max" not in source
