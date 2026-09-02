from pathlib import Path
import json
import cv2

ROOT = Path(__file__).resolve().parents[1]


def _load():
    template = json.loads(
        (ROOT / "templates" / "jee.json").read_text(encoding="utf-8")
    )
    image = cv2.imread(
        str(ROOT / "references" / template["reference_image"])
    )
    assert image is not None
    return template, image


def _fill(image, x, y):
    cv2.circle(
        image,
        (int(round(x)), int(round(y))),
        7,
        (30, 30, 30),
        -1,
        lineType=cv2.LINE_AA,
    )


def _decimal_positions(template, question_number):
    layout = template["numerical_layout"]
    for section in template["numerical_sections"]:
        start = int(section["start_question"])
        xs = section["question_x_positions"]
        if start <= question_number < start + len(xs):
            base_x = int(xs[question_number - start])
            return [
                (
                    int(after_column),
                    base_x + int(offset),
                    int(section["decimal_y"]),
                )
                for after_column, offset in zip(
                    layout["decimal_after_columns"],
                    layout["decimal_offsets"],
                )
            ]
    raise AssertionError("question not found")


def test_blank_decimal_row_has_no_false_fills():
    from jee_reader import scan_jee_numerical_sections_robust

    template, image = _load()
    detected, _debug = scan_jee_numerical_sections_robust(image, template)

    for question in (71, 72, 74):
        filled = [
            item["after_column"]
            for item in detected[question]["decimal_points"]
            if item.get("filled")
        ]
        assert filled == []


def test_71_first_and_72_third_decimal_when_marked():
    from jee_reader import (
        scan_jee_numerical_sections_robust,
    )

    template, image = _load()

    expected = {
        71: [1],
        72: [3],
    }

    for question, wanted in expected.items():
        positions = {
            after: (x, y)
            for after, x, y
            in _decimal_positions(
                template,
                question,
            )
        }

        for after in wanted:
            x, y = positions[after]

            _fill(
                image,
                x,
                y,
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
                item["after_column"]
            )
            for item
            in detected[question]["decimal_points"]
            if item.get("filled")
        )

        assert filled == wanted

def test_74_multiple_real_decimal_fills_remain_detected():
    from jee_reader import scan_jee_numerical_sections_robust

    template, image = _load()
    positions = {
        after: (x, y)
        for after, x, y in _decimal_positions(template, 74)
    }

    for after in (2, 5):
        x, y = positions[after]
        _fill(image, x, y)

    detected, _debug = scan_jee_numerical_sections_robust(image, template)
    record = detected[74]

    filled = sorted(
        int(item["after_column"])
        for item in record["decimal_points"]
        if item.get("filled")
    )

    assert filled == [2, 5]
    assert record["decimal_status"] == "MULTIPLE"
    assert record["answer"] == "UNCERTAIN"


def test_v6_2_contract():
    reader = (ROOT / "jee_reader.py").read_text(encoding="utf-8")
    scanner = (ROOT / "scanner.py").read_text(encoding="utf-8")
    template = json.loads(
        (ROOT / "templates" / "jee.json").read_text(encoding="utf-8")
    )

    # v6.3 supersedes the v6.2 exact-centre decimal reader.
    assert "decimal_annulus_fill_v6_3" in reader
    assert "individual_hough_center_v6_3" in reader
    assert template["jee_numeric_decimal_local_search"] == 0
    assert "decimal_color = (0, 255, 0)" in scanner
