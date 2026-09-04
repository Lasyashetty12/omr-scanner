from pathlib import Path

import cv2
import numpy as np

import scanner

ROOT = Path(__file__).resolve().parents[1]


def test_camera_profile_uses_inner_core_and_small_search():
    template = {
        "bubble_radius": 10,
        "blank_threshold": 0.72,
        "filled_threshold": 0.75,
        "multiple_threshold": 0.75,
        "mcq_sections": [
            {
                "start_question": 1,
                "total_questions": 1,
                "option_x": {
                    "A": 100,
                    "B": 130,
                    "C": 160,
                    "D": 190,
                },
                "question_y_positions": [100],
            }
        ],
    }

    profile = scanner.build_jee_camera_mcq_template(
        template
    )

    assert profile["bubble_radius"] == 8
    assert profile["mcq_sections"][0]["search_radius"] == 2
    assert profile["mcq_sections"][0]["search_step"] == 1

    assert template["bubble_radius"] == 10
    assert "search_radius" not in template["mcq_sections"][0]

    assert profile["blank_threshold"] == 0.72
    assert profile["filled_threshold"] == 0.75
    assert profile["multiple_threshold"] == 0.75


def test_camera_preprocess_and_profile_preserve_image_dimensions():
    image = np.full(
        (420, 300, 3),
        235,
        dtype=np.uint8,
    )

    result = scanner.prepare_jee_camera_mcq_image(
        image
    )

    assert result.shape == image.shape


def test_inner_core_separates_blank_ring_from_filled_bubble():
    image = np.full(
        (420, 300, 3),
        235,
        dtype=np.uint8,
    )

    cv2.circle(
        image,
        (90, 190),
        10,
        (45, 45, 45),
        1,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        image,
        "A",
        (87, 193),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.20,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )

    cv2.circle(
        image,
        (200, 190),
        9,
        (45, 45, 45),
        -1,
        lineType=cv2.LINE_AA,
    )

    processed = scanner.prepare_jee_camera_mcq_image(
        image
    )

    gray = scanner.normalize_grayscale(
        processed
    )

    profile = {
        "bubble_radius": 8,
        "dark_pixel_threshold": 120,
    }

    blank_score = scanner.get_fill_ratio(
        gray,
        90,
        190,
        profile,
    )

    filled_score = scanner.get_fill_ratio(
        gray,
        200,
        190,
        profile,
    )

    assert blank_score < 0.30
    assert filled_score > 0.75
    assert filled_score - blank_score > 0.45


def test_camera_path_uses_camera_profile_only():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(
        encoding="utf-8"
    )

    process_start = source.index(
        "def process_omr("
    )

    process_source = source[
        process_start:
    ]

    jee_start = process_source.index(
        'elif exam_name == "JEE":'
    )

    jee_end = process_source.index(
        "\n    else:",
        jee_start,
    )

    branch = process_source[
        jee_start:jee_end
    ]

    assert "if camera_capture:" in branch
    assert "build_jee_camera_mcq_template(" in branch
    assert "camera_mcq_template" in branch

    assert (
        "scan_jee_mcq_sections(\n"
        "                    camera_mcq_image,\n"
        "                    camera_mcq_template,"
        in branch
    )

    assert (
        "scan_jee_numerical_sections_robust(\n"
        "                recognition_image,"
        in branch
    )
