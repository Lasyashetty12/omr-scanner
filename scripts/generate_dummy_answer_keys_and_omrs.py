"""Generate deterministic dummy answer keys and matching filled test OMRs.

These assets are for scanner verification only. They are not real exam keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
SERIES = ("P", "Q", "R", "S")
OPTIONS = ("A", "B", "C", "D")
TEST_ASSET_DIR = ROOT / "test_assets"

JEE_MCQ_QUESTIONS = (
    *range(1, 21),
    *range(26, 46),
    *range(51, 71),
)
JEE_NUMERICAL_QUESTIONS = (
    *range(21, 26),
    *range(46, 51),
    *range(71, 76),
)
JEE_NUMERICAL_VALUES = (
    "-1.2",
    "3.4",
    "5.6",
    "7.8",
    "9.0",
    "-2.3",
    "4.5",
    "6.7",
    "8.9",
    "1.0",
    "-3.4",
    "5.7",
    "7.1",
    "9.2",
    "2.6",
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def option_for(question: int, series: str) -> str:
    return OPTIONS[(question - 1 + SERIES.index(series)) % len(OPTIONS)]


def common_metadata(exam: str, series: str):
    return {
        "exam": exam,
        "series": series,
        "dummy": True,
        "warning": "TEST ONLY - replace with the official answer key before production use",
    }


def generate_answer_keys():
    for exam in ("NEET", "KCET"):
        marking = (
            {"correct": 4, "wrong": -1, "blank": 0, "multiple": -1}
            if exam == "NEET"
            else {"correct": 1, "wrong": 0, "blank": 0, "multiple": 0}
        )
        for series in SERIES:
            key = {
                **common_metadata(exam, series),
                "paper_code": series,
                "marking": marking,
                "answers": {
                    str(question): option_for(question, series)
                    for question in range(1, 209)
                },
            }
            write_json(ROOT / "answer_keys" / exam.lower() / f"{series}.json", key)

    for series in SERIES:
        offset = SERIES.index(series)
        rotated_values = JEE_NUMERICAL_VALUES[offset:] + JEE_NUMERICAL_VALUES[:offset]
        key = {
            **common_metadata("JEE", series),
            "marking": {
                "mcq_correct": 4,
                "mcq_wrong": -1,
                "mcq_blank": 0,
                "mcq_multiple": -1,
                "numerical_correct": 4,
                "numerical_wrong": 0,
                "numerical_blank": 0,
            },
            "answers": {
                "mcq": {
                    str(question): option_for(question, series)
                    for question in JEE_MCQ_QUESTIONS
                },
                "numerical": {
                    str(question): value
                    for question, value in zip(JEE_NUMERICAL_QUESTIONS, rotated_values)
                },
            },
        }
        write_json(ROOT / "answer_keys" / "jee" / f"{series}.json", key)


def fill_circle(image, point, radius=8):
    cv2.circle(image, tuple(map(int, point)), radius, (0, 0, 0), -1)


def generate_neet_kcet_test_omr(series="P"):
    template = load_json(ROOT / "templates" / "neet.json")
    key = load_json(ROOT / "answer_keys" / "neet" / f"{series}.json")
    image = cv2.imread(str(ROOT / "references" / template["reference_image"]))
    if image is None:
        raise RuntimeError("Could not load the NEET/KCET reference image.")

    fill_circle(image, template["series"]["coordinates"][series])
    per_column = int(template["questions_per_column"])
    for question_text, option in key["answers"].items():
        question = int(question_text)
        column_index = (question - 1) // per_column
        row_index = (question - 1) % per_column
        point = (
            template["columns"][column_index][option],
            template["question_y_positions"][row_index],
        )
        fill_circle(image, point)

    output = TEST_ASSET_DIR / f"filled_neet_kcet_series_{series}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Could not write {output}.")
    return output


def _jee_mcq_points(template):
    points = {}
    for section in template["mcq_sections"]:
        start = int(section["start_question"])
        for index, y in enumerate(section["question_y_positions"]):
            question = start + index
            points[question] = {
                option: (x, y)
                for option, x in section["option_x"].items()
            }
    return points


def _jee_numerical_configs(template):
    layout = template["numerical_layout"]
    configs = {}
    for section in template["numerical_sections"]:
        start = int(section["start_question"])
        for index, base_x in enumerate(section["question_x_positions"]):
            configs[start + index] = {
                "base_x": int(base_x),
                "digit_offsets": layout["digit_offsets"],
                "digit_y_positions": section["digit_y_positions"],
                "decimal_offsets": layout["decimal_offsets"],
                "decimal_after_columns": layout["decimal_after_columns"],
                "decimal_y": int(section["decimal_y"]),
                "sign_x": int(base_x) + int(layout.get("sign_offset", 0)),
                "sign_y": int(section["sign_y"]),
            }
    return configs


def _fill_numerical_value(image, config, value):
    text = str(value)
    if text.startswith("-"):
        fill_circle(image, (config["sign_x"], config["sign_y"]))
        text = text[1:]

    if "." in text:
        integer, fraction = text.split(".", 1)
        digits = integer + fraction
        after_column = len(integer)
        decimal_index = config["decimal_after_columns"].index(after_column)
        fill_circle(
            image,
            (
                config["base_x"] + config["decimal_offsets"][decimal_index],
                config["decimal_y"],
            ),
        )
    else:
        digits = text

    if len(digits) > len(config["digit_offsets"]):
        raise ValueError(f"Dummy numerical value {value} exceeds the OMR columns.")

    for column_index, digit in enumerate(digits):
        fill_circle(
            image,
            (
                config["base_x"] + config["digit_offsets"][column_index],
                config["digit_y_positions"][int(digit)],
            ),
        )


def generate_jee_test_omr(series="P"):
    template = load_json(ROOT / "templates" / "jee.json")
    key = load_json(ROOT / "answer_keys" / "jee" / f"{series}.json")
    image = cv2.imread(str(ROOT / "references" / template["reference_image"]))
    if image is None:
        raise RuntimeError("Could not load the JEE reference image.")

    fill_circle(image, template["series"]["coordinates"][series])
    mcq_points = _jee_mcq_points(template)
    for question_text, option in key["answers"]["mcq"].items():
        fill_circle(image, mcq_points[int(question_text)][option])

    numerical_configs = _jee_numerical_configs(template)
    for question_text, value in key["answers"]["numerical"].items():
        _fill_numerical_value(image, numerical_configs[int(question_text)], value)

    output = TEST_ASSET_DIR / f"filled_jee_series_{series}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Could not write {output}.")
    return output


def main():
    generate_answer_keys()
    outputs = (
        generate_neet_kcet_test_omr("P"),
        generate_jee_test_omr("P"),
    )
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
