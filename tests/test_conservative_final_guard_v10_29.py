from pathlib import Path

import cv2
import numpy as np

from ml_omr.final_guard_v10_29 import (
    apply_strict_ml_blank_veto,
    detect_series_cv_fallback,
    resolve_strict_jee_secondary_multiple,
)


def _option(
    *,
    filled,
    blank,
    disk,
    darkness,
    core=0.8,
):
    return {
        "ml_filled_probability": float(filled),
        "ml_blank_probability": float(blank),
        "ml_ambiguous_probability": max(
            0.0,
            1.0 - float(filled) - float(blank),
        ),
        "metrics": {
            "disk_dark_ratio": float(disk),
            "center_darkness": float(darkness),
            "core_dark_ratio": float(core),
        },
    }


def test_strict_blank_veto_removes_only_decisive_false_single():
    option_data = {
        "A": _option(
            filled=0.02,
            blank=0.97,
            disk=0.42,
            darkness=58,
        ),
        "B": _option(
            filled=0.01,
            blank=0.98,
            disk=0.30,
            darkness=40,
        ),
        "C": _option(
            filled=0.01,
            blank=0.98,
            disk=0.28,
            darkness=38,
        ),
        "D": _option(
            filled=0.01,
            blank=0.98,
            disk=0.26,
            darkness=36,
        ),
    }

    result = apply_strict_ml_blank_veto(
        decision={
            "answer": "A",
            "status": "answered",
            "multiple_options": [],
        },
        option_data=option_data,
        questions_per_column=60,
    )

    assert result["status"] == "blank"
    assert result["answer"] is None
    assert result["strict_ml_blank_veto"] is True


def test_real_fill_is_never_removed_by_blank_veto():
    option_data = {
        "A": _option(
            filled=0.93,
            blank=0.03,
            disk=0.88,
            darkness=130,
        ),
        "B": _option(
            filled=0.02,
            blank=0.97,
            disk=0.30,
            darkness=40,
        ),
        "C": _option(
            filled=0.02,
            blank=0.97,
            disk=0.30,
            darkness=40,
        ),
        "D": _option(
            filled=0.02,
            blank=0.97,
            disk=0.30,
            darkness=40,
        ),
    }

    original = {
        "answer": "A",
        "status": "answered",
        "multiple_options": [],
    }

    result = apply_strict_ml_blank_veto(
        decision=original,
        option_data=option_data,
        questions_per_column=60,
    )

    assert result == original


def test_strict_jee_secondary_multiple_requires_exactly_one_extra_fill():
    option_data = {
        "A": _option(
            filled=0.94,
            blank=0.03,
            disk=0.88,
            darkness=125,
        ),
        "B": _option(
            filled=0.03,
            blank=0.94,
            disk=0.31,
            darkness=42,
        ),
        "C": _option(
            filled=0.78,
            blank=0.10,
            disk=0.76,
            darkness=96,
        ),
        "D": _option(
            filled=0.02,
            blank=0.96,
            disk=0.30,
            darkness=40,
        ),
    }

    answer, decision = (
        resolve_strict_jee_secondary_multiple(
            stable_answer="A",
            ml_answer="A",
            ml_decision={
                "answer": "A",
                "status": "answered",
                "options": option_data,
            },
        )
    )

    assert answer == "MULTIPLE"
    assert decision["multiple_options"] == [
        "A",
        "C",
    ]


def test_series_cv_fallback_detects_solid_q_bubble():
    image = np.full(
        (420, 560),
        230,
        dtype=np.uint8,
    )

    coordinates = {
        "P": [100, 210],
        "Q": [210, 210],
        "R": [320, 210],
        "S": [430, 210],
    }

    for point in coordinates.values():
        cv2.circle(
            image,
            tuple(point),
            11,
            85,
            2,
            lineType=cv2.LINE_AA,
        )

    cv2.circle(
        image,
        tuple(coordinates["Q"]),
        8,
        30,
        -1,
        lineType=cv2.LINE_AA,
    )

    image = cv2.GaussianBlur(
        image,
        (3, 3),
        0,
    )

    result = detect_series_cv_fallback(
        image,
        {
            "exam_name": "KCET",
            "series": {
                "coordinates": coordinates,
            },
        },
        exam_name="KCET",
    )

    assert result is not None
    assert result["value"] == "Q"


def test_install_targets_exist_after_script():
    root = Path(__file__).resolve().parents[1]
    hybrid = (
        root
        / "ml_omr"
        / "hybrid_reader.py"
    ).read_text(encoding="utf-8")
    scanner = (
        root
        / "scanner.py"
    ).read_text(encoding="utf-8")

    assert "apply_strict_ml_blank_veto(" in hybrid
    assert "_detect_exam_series_legacy(" in scanner
    assert "resolve_strict_jee_secondary_multiple(" in scanner
