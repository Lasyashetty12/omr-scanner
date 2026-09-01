from pathlib import Path
import json
import cv2

from identity_reader import detect_identity_fields

ROOT = Path(__file__).resolve().parents[1]


def _fill(image, center, radius=7):
    cv2.circle(
        image,
        (int(center[0]), int(center[1])),
        int(radius),
        (0, 0, 0),
        -1,
        lineType=cv2.LINE_AA,
    )


def test_jee_roll_number_reader():
    image = cv2.imread(str(ROOT / "references" / "jee_generated.png"))
    assert image is not None

    template = json.loads(
        (ROOT / "templates" / "jee.json").read_text(encoding="utf-8")
    )
    config = template["identity"]["roll_number"]

    expected = "2471963"
    for column_index, digit in enumerate(expected):
        _fill(
            image,
            (
                config["x_positions"][column_index],
                config["y_positions"][int(digit)],
            ),
        )

    result = detect_identity_fields(image, template)

    assert result["roll_number"] == expected
    assert result["roll_number_details"]["complete"] is True


def test_kcet_roll_class_and_exam_reader():
    image = cv2.imread(str(ROOT / "references" / "neet_kcet_generated.png"))
    assert image is not None

    template = json.loads(
        (ROOT / "templates" / "kcet.json").read_text(encoding="utf-8")
    )
    identity = template["identity"]

    expected_roll = "24719638"
    roll = identity["roll_number"]

    for column_index, digit in enumerate(expected_roll):
        _fill(
            image,
            (
                roll["x_positions"][column_index],
                roll["y_positions"][int(digit)],
            ),
        )

    _fill(image, identity["class"]["choices"]["II"])
    _fill(image, identity["exam"]["choices"]["KCET"])

    result = detect_identity_fields(image, template)

    assert result["roll_number"] == expected_roll
    assert result["class"] == "II"
    assert result["exam"] == "KCET"
