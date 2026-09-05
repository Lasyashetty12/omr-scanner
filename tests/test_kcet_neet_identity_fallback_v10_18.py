import json
from pathlib import Path

import cv2
import numpy as np

from identity_reader import _detect_choice_row
from scanner import merge_identity_fallback


ROOT = Path(__file__).resolve().parents[1]


def load_kcet_template():
    return json.loads(
        (ROOT / "templates" / "kcet.json").read_text(encoding="utf-8")
    )


def test_exam_choice_row_calibrates_mobile_vertical_offset():
    template = load_kcet_template()
    config = template["identity"]["exam"]
    image = np.full((2200, 1600), 255, dtype=np.uint8)
    vertical_offset = 12

    for x, y in config["choices"].values():
        cv2.circle(
            image,
            (int(x), int(y + vertical_offset)),
            11,
            0,
            2,
        )

    kcet_x, kcet_y = config["choices"]["KCET"]
    cv2.circle(
        image,
        (int(kcet_x), int(kcet_y + vertical_offset)),
        8,
        0,
        -1,
    )

    result = _detect_choice_row(image, config, template)

    assert result["value"] == "KCET"
    assert result["y_calibrated"] is True
    assert result["sampling_y"] == kcet_y + vertical_offset


def test_corrected_image_fallback_recovers_only_missing_identity_fields():
    primary = {
        "roll_number": "40316817",
        "roll_number_details": {"reader": "primary"},
        "class": "II",
        "class_details": {"best_score": 0.91},
        "exam": None,
        "exam_details": {"best_score": 0.45},
    }
    fallback = {
        "roll_number": "99999999",
        "roll_number_details": {"reader": "fallback"},
        "class": "I",
        "class_details": {"best_score": 0.99},
        "exam": "KCET",
        "exam_details": {"best_score": 1.0},
    }

    merged = merge_identity_fallback(primary, fallback)

    assert merged["roll_number"] == "40316817"
    assert merged["roll_number_details"] == {"reader": "primary"}
    assert merged["class"] == "II"
    assert merged["exam"] == "KCET"
    assert merged["exam_details"] == {"best_score": 1.0}
    assert merged["fallback_source"] == "corrected_image"
    assert merged["fallback_recovered"] == ["exam"]


def test_kcet_neet_pipeline_retries_identity_on_unenhanced_corrected_image():
    source = (ROOT / "scanner.py").read_text(encoding="utf-8")

    assert 'template_exam_name in ("NEET", "KCET")' in source
    assert "corrected_identity = detect_identity_fields(" in source
    assert "identity = merge_identity_fallback(" in source
