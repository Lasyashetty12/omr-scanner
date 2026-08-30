import json
from pathlib import Path

import cv2

from omr_preprocess.registration_align import (
    canonicalize_omr,
    detect_registration_blocks,
)
from scanner import (
    detect_exam_series,
    detect_jee_series,
    normalize_grayscale,
    scan_jee_answers,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_template(name):
    return json.loads((ROOT / "templates" / f"{name}.json").read_text())


def test_generated_neet_and_kcet_share_the_measured_208_row_grid():
    neet = _load_template("neet")
    kcet = _load_template("kcet")

    for template in (neet, kcet):
        assert template["sheet_width"] == 1600
        assert template["sheet_height"] == 2263
        assert template["total_questions"] == 208
        assert template["questions_per_column"] == 52
        assert len(template["columns"]) == 4
        assert len(template["question_y_positions"]) == 52
        assert template["question_y_positions"][0] == 338
        assert template["question_y_positions"][-1] == 1966


def test_reference_registration_blocks_are_the_actual_corner_boxes():
    expected = {
        "neet_kcet_generated.png": ((95, 107), (1512, 106), (1513, 2168), (85, 2167)),
        "jee_generated.png": ((51, 44), (1549, 44), (1549, 2213), (51, 2214)),
    }

    for filename, centers in expected.items():
        image = cv2.imread(str(ROOT / "references" / filename))
        markers, _ = detect_registration_blocks(image)
        for detected, measured in zip(markers, centers):
            assert abs(float(detected[0]) - measured[0]) <= 3
            assert abs(float(detected[1]) - measured[1]) <= 3


def test_rotated_generated_jee_sheet_is_registered_upright():
    reference_path = ROOT / "references" / "jee_generated.png"
    reference = cv2.imread(str(reference_path))
    rotated = cv2.rotate(reference, cv2.ROTATE_90_CLOCKWISE)

    corrected, debug = canonicalize_omr(
        rotated,
        reference_path,
        output_size=(1600, 2263),
        use_orb=True,
        use_ecc=True,
    )

    assert corrected.shape[:2] == (2263, 1600)
    assert debug["orientation"]["selected_rotation"] == 90
    assert debug["ecc_score"] > 0.99


def test_generated_jee_bubbles_read_mcq_signed_decimal_and_series(tmp_path):
    template = _load_template("jee")
    image = cv2.imread(str(ROOT / "references" / "jee_generated.png"))

    # P series; q1=C; q70=D; q21=-1.2 with decimal after digit column 1.
    for point in (
        (368, 186),
        (546, 426),
        (1474, 738),
        (182, 932),
        (208, 962),
        (208, 864),
        (182, 1192),
    ):
        cv2.circle(image, point, 8, (0, 0, 0), -1)

    gray = normalize_grayscale(image)
    series = detect_jee_series(gray, template)
    answers = scan_jee_answers(image, template)

    assert series["value"] == "P"
    assert answers["mcq"][1]["answer"] == "C"
    assert answers["mcq"][70]["answer"] == "D"
    assert answers["numerical"][21]["answer"] == "-1.2"
    assert answers["numerical"][22]["answer"] == "BLANK"


def test_kcet_series_tolerates_small_mobile_alignment_offset():
    template = _load_template("kcet")
    image = cv2.imread(str(ROOT / "references" / "neet_kcet_generated.png"))
    series_x, series_y = template["series"]["coordinates"]["Q"]
    cv2.circle(image, (series_x + 5, series_y - 4), 8, (0, 0, 0), -1)

    detected = detect_exam_series(
        normalize_grayscale(image),
        template,
        exam_name="KCET",
    )

    assert detected["value"] == "Q"
    assert detected["sampling_centres"]["Q"] != [series_x, series_y]
