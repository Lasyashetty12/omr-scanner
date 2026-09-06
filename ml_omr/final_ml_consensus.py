from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np


PROFILE = "final_ml_consensus_v10_28"
_REFERENCE_CACHE: Dict[str, np.ndarray] = {}


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _extract_square(gray, x, y, radius):
    x = int(round(float(x)))
    y = int(round(float(y)))
    radius = int(radius)

    x0 = x - radius
    y0 = y - radius
    x1 = x + radius + 1
    y1 = y + radius + 1

    if (
        x0 < 0
        or y0 < 0
        or x1 > gray.shape[1]
        or y1 > gray.shape[0]
    ):
        return None

    return gray[y0:y1, x0:x1].astype(np.float32)


def _reference_gray(template):
    reference_name = str(
        template.get(
            "reference_image",
            "neet_kcet_generated.png",
        )
    )

    if reference_name in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[reference_name]

    path = (
        Path(__file__).resolve().parents[1]
        / "references"
        / reference_name
    )

    if not path.exists():
        return None

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    expected_width = int(
        template.get("sheet_width", image.shape[1])
    )
    expected_height = int(
        template.get("sheet_height", image.shape[0])
    )

    if (
        image.shape[1] != expected_width
        or image.shape[0] != expected_height
    ):
        image = cv2.resize(
            image,
            (expected_width, expected_height),
            interpolation=cv2.INTER_LINEAR,
        )

    _REFERENCE_CACHE[reference_name] = image
    return image


def _reference_center(question, option, template):
    qpc = int(template.get("questions_per_column", 60))
    columns = list(template.get("columns", []))
    y_positions = list(
        template.get("question_y_positions", [])
    )

    if qpc <= 0 or not columns or not y_positions:
        return None

    index = int(question) - 1
    column_index = index // qpc
    row_index = index % qpc

    if (
        column_index < 0
        or column_index >= len(columns)
        or row_index >= len(y_positions)
    ):
        return None

    column = columns[column_index]

    if option not in column:
        return None

    return (
        float(column[option]),
        float(y_positions[row_index]),
    )


def _reference_extra_ink(
    gray,
    reference,
    actual_center,
    reference_center,
):
    """
    Compare the photographed bubble with the blank generated OMR.
    Printed rings occur in both images and mostly subtract away; new pen ink
    remains as positive extra-dark evidence.
    """
    if reference is None or reference_center is None:
        return {
            "score": 0.0,
            "available": False,
        }

    radius = 13
    size = radius * 2 + 1

    yy, xx = np.ogrid[:size, :size]
    centre = radius
    distance_sq = (
        (xx - centre) ** 2
        + (yy - centre) ** 2
    )

    core_mask = distance_sq <= 5 * 5
    disk_mask = distance_sq <= 8 * 8
    annulus_mask = (
        (distance_sq >= 10 * 10)
        & (distance_sq <= 13 * 13)
    )

    best = None

    for actual_dy in range(-2, 3):
        for actual_dx in range(-2, 3):
            current = _extract_square(
                gray,
                actual_center[0] + actual_dx,
                actual_center[1] + actual_dy,
                radius,
            )

            if current is None:
                continue

            current_annulus = current[annulus_mask]

            if current_annulus.size < 30:
                continue

            for ref_dy in range(-1, 2):
                for ref_dx in range(-1, 2):
                    blank = _extract_square(
                        reference,
                        reference_center[0] + ref_dx,
                        reference_center[1] + ref_dy,
                        radius,
                    )

                    if blank is None:
                        continue

                    blank_annulus = blank[annulus_mask]

                    if blank_annulus.size < 30:
                        continue

                    brightness_shift = float(
                        np.median(blank_annulus)
                        - np.median(current_annulus)
                    )

                    adjusted = np.clip(
                        current + brightness_shift,
                        0.0,
                        255.0,
                    )

                    annulus_error = float(
                        np.mean(
                            np.abs(
                                blank[annulus_mask]
                                - adjusted[annulus_mask]
                            )
                        )
                    )

                    if (
                        best is None
                        or annulus_error < best["annulus_error"]
                    ):
                        best = {
                            "blank": blank,
                            "current": adjusted,
                            "annulus_error": annulus_error,
                            "dx": actual_dx,
                            "dy": actual_dy,
                        }

    if best is None:
        return {
            "score": 0.0,
            "available": False,
        }

    extra_dark = np.clip(
        best["blank"] - best["current"],
        0.0,
        255.0,
    )

    core_extra = extra_dark[core_mask]
    disk_extra = extra_dark[disk_mask]

    core_mean = float(np.mean(core_extra))
    disk_mean = float(np.mean(disk_extra))

    core_fraction = float(
        np.mean(core_extra >= 30.0)
    )
    disk_fraction = float(
        np.mean(disk_extra >= 26.0)
    )

    score = (
        0.30 * _clip01(core_mean / 105.0)
        + 0.25 * core_fraction
        + 0.25 * _clip01(disk_mean / 85.0)
        + 0.20 * disk_fraction
    )

    return {
        "score": round(_clip01(score), 4),
        "available": True,
        "core_extra_mean": round(core_mean, 3),
        "disk_extra_mean": round(disk_mean, 3),
        "core_extra_fraction": round(core_fraction, 4),
        "disk_extra_fraction": round(disk_fraction, 4),
        "annulus_error": round(
            float(best["annulus_error"]),
            3,
        ),
        "dx": int(best["dx"]),
        "dy": int(best["dy"]),
    }


def _solid_center_evidence(
    gray,
    center_x,
    center_y,
    search_radius=3,
):
    """
    Measure solid dark ink through the physical bubble centre.
    Empty printed outlines stay relatively light in the core.
    """
    gray = _gray(gray)

    radius = 14
    height, width = gray.shape[:2]

    yy, xx = np.ogrid[
        -radius:radius + 1,
        -radius:radius + 1,
    ]

    distance_sq = xx * xx + yy * yy
    core_mask = distance_sq <= 4 * 4
    disk_mask = distance_sq <= 7 * 7
    paper_mask = (
        (distance_sq >= 11 * 11)
        & (distance_sq <= 14 * 14)
    )

    best = None

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            cx = int(round(float(center_x))) + dx
            cy = int(round(float(center_y))) + dy

            if (
                cx - radius < 0
                or cy - radius < 0
                or cx + radius >= width
                or cy + radius >= height
            ):
                continue

            patch = gray[
                cy - radius:cy + radius + 1,
                cx - radius:cx + radius + 1,
            ]

            core_pixels = patch[core_mask].astype(np.float32)
            disk_pixels = patch[disk_mask].astype(np.float32)
            paper_pixels = patch[paper_mask].astype(np.float32)

            if (
                core_pixels.size < 20
                or disk_pixels.size < 60
                or paper_pixels.size < 80
            ):
                continue

            paper_level = float(
                np.percentile(paper_pixels, 75.0)
            )

            dark_threshold = float(
                np.clip(
                    paper_level - 38.0,
                    70.0,
                    190.0,
                )
            )

            core_mean = float(np.mean(core_pixels))
            core_p90 = float(
                np.percentile(core_pixels, 90.0)
            )
            disk_mean = float(np.mean(disk_pixels))

            core_delta = paper_level - core_mean
            p90_delta = paper_level - core_p90
            disk_delta = paper_level - disk_mean

            core_dark_ratio = float(
                np.mean(core_pixels < dark_threshold)
            )
            disk_dark_ratio = float(
                np.mean(disk_pixels < dark_threshold)
            )

            score = (
                0.29
                * _clip01((core_delta - 18.0) / 95.0)
                + 0.23
                * _clip01((p90_delta - 8.0) / 80.0)
                + 0.18
                * _clip01((disk_delta - 15.0) / 85.0)
                + 0.17 * core_dark_ratio
                + 0.13 * disk_dark_ratio
                - 0.012 * (abs(dx) + abs(dy))
            )

            record = {
                "score": _clip01(score),
                "core_delta": core_delta,
                "p90_delta": p90_delta,
                "disk_delta": disk_delta,
                "core_dark_ratio": core_dark_ratio,
                "disk_dark_ratio": disk_dark_ratio,
                "paper_level": paper_level,
                "dx": dx,
                "dy": dy,
            }

            if (
                best is None
                or record["score"] > best["score"]
            ):
                best = record

    if best is None:
        return {
            "score": 0.0,
        }

    result = {}

    for key, value in best.items():
        if key in ("dx", "dy"):
            result[key] = int(value)
        else:
            result[key] = round(float(value), 4)

    return result


def _ml_probability(option_debug, label):
    direct_key = "ml_" + label + "_probability"

    if direct_key in option_debug:
        return float(option_debug.get(direct_key, 0.0))

    prediction = option_debug.get("ml", {})

    if not isinstance(prediction, dict):
        return 0.0

    probabilities = prediction.get("probabilities", {})

    if not isinstance(probabilities, dict):
        return 0.0

    return float(probabilities.get(label, 0.0))


def _option_consensus(
    ml_filled,
    ml_blank,
    ml_ambiguous,
    reference_score,
    solid_score,
):
    # ONNX model is the main signal. The other two are image-only validators.
    combined = (
        0.60 * ml_filled
        + 0.24 * reference_score
        + 0.16 * solid_score
        - 0.08 * ml_blank
        - 0.03 * ml_ambiguous
    )

    combined = _clip01(combined)

    visual_support = (
        reference_score >= 0.14
        or solid_score >= 0.50
    )

    strong_model = (
        ml_filled >= 0.78
        and ml_filled >= ml_blank + 0.16
    )

    verified_fill = (
        (strong_model and visual_support)
        or (
            ml_filled >= 0.60
            and reference_score >= 0.22
            and solid_score >= 0.48
        )
        or (
            reference_score >= 0.42
            and solid_score >= 0.68
            and ml_blank < 0.62
        )
    )

    probable_fill = (
        (
            combined >= 0.54
            and ml_filled >= 0.46
            and visual_support
        )
        or (
            reference_score >= 0.30
            and solid_score >= 0.58
            and ml_filled >= 0.35
        )
    )

    verified_blank = (
        (
            ml_blank >= 0.72
            and ml_filled <= 0.30
            and reference_score < 0.13
            and solid_score < 0.48
        )
        or (
            ml_filled < 0.16
            and reference_score < 0.09
            and solid_score < 0.40
        )
    )

    return {
        "combined": round(combined, 4),
        "verified_fill": bool(verified_fill),
        "probable_fill": bool(probable_fill),
        "verified_blank": bool(verified_blank),
    }


def refine_neet_kcet_answers_with_ml(
    *,
    gray,
    coordinates,
    template,
    raw_answers,
    ml_debug,
    reference_gray=None,
):
    """
    Final NEET/KCET model-primary filled/blank pass.

    Uses the existing ONNX filled/blank/ambiguous probabilities as the main
    signal, then verifies them against:
      - extra ink versus the blank generated OMR,
      - solid central ink at the CV-fitted physical bubble.
    """
    exam_name = str(
        template.get("exam_name", "")
    ).strip().upper()

    if exam_name not in ("NEET", "KCET"):
        return raw_answers, ml_debug

    gray = _gray(gray)

    if reference_gray is None:
        reference_gray = _reference_gray(template)

    options = list(
        template.get(
            "options",
            ["A", "B", "C", "D"],
        )
    )

    final_answers = dict(raw_answers)
    final_debug = dict(ml_debug)

    for question_key, option_map in coordinates.items():
        try:
            question = int(question_key)
        except (TypeError, ValueError):
            continue

        existing_debug = final_debug.get(
            question_key,
            final_debug.get(question, {}),
        )

        if not isinstance(existing_debug, dict):
            existing_debug = {}

        existing_options = existing_debug.get(
            "options",
            {},
        )

        option_results = {}

        for option in options:
            if option not in option_map:
                continue

            option_debug = (
                existing_options.get(option, {})
                if isinstance(existing_options, dict)
                else {}
            )

            ml_filled = _ml_probability(
                option_debug,
                "filled",
            )
            ml_blank = _ml_probability(
                option_debug,
                "blank",
            )
            ml_ambiguous = _ml_probability(
                option_debug,
                "ambiguous",
            )

            actual_center = option_map[option]
            ref_center = _reference_center(
                question,
                option,
                template,
            )

            reference_result = _reference_extra_ink(
                gray,
                reference_gray,
                (
                    float(actual_center[0]),
                    float(actual_center[1]),
                ),
                ref_center,
            )

            solid_result = _solid_center_evidence(
                gray,
                float(actual_center[0]),
                float(actual_center[1]),
                search_radius=3,
            )

            consensus = _option_consensus(
                ml_filled,
                ml_blank,
                ml_ambiguous,
                float(reference_result.get("score", 0.0)),
                float(solid_result.get("score", 0.0)),
            )

            option_results[option] = {
                "ml_filled": round(ml_filled, 4),
                "ml_blank": round(ml_blank, 4),
                "ml_ambiguous": round(ml_ambiguous, 4),
                "reference": reference_result,
                "solid": solid_result,
                **consensus,
            }

        if len(option_results) < 2:
            continue

        ranked = sorted(
            option_results,
            key=lambda option:
                float(
                    option_results[option]["combined"]
                ),
            reverse=True,
        )

        verified = [
            option
            for option in ranked
            if option_results[option]["verified_fill"]
        ]

        probable = [
            option
            for option in ranked
            if option_results[option]["probable_fill"]
        ]

        supported = [
            option
            for option in ranked
            if option in verified or option in probable
        ]

        old_answer = final_answers.get(
            question_key,
            final_answers.get(question),
        )

        answer = None
        status = "ambiguous"
        multiple_options = []

        if len(verified) >= 2:
            answer = "MULTIPLE"
            status = "multiple"
            multiple_options = verified

        elif (
            len(verified) == 1
            and len(supported) >= 2
        ):
            second = [
                option
                for option in supported
                if option != verified[0]
            ][0]

            second_data = option_results[second]

            if (
                second_data["ml_filled"] >= 0.52
                and (
                    float(
                        second_data["reference"].get(
                            "score",
                            0.0,
                        )
                    )
                    >= 0.18
                    or float(
                        second_data["solid"].get(
                            "score",
                            0.0,
                        )
                    )
                    >= 0.55
                )
            ):
                answer = "MULTIPLE"
                status = "multiple"
                multiple_options = [
                    verified[0],
                    second,
                ]
            else:
                answer = verified[0]
                status = "answered"

        elif len(verified) == 1:
            answer = verified[0]
            status = "answered"

        elif len(probable) == 1:
            candidate = probable[0]

            second_score = float(
                option_results[ranked[1]]["combined"]
            )

            if (
                float(
                    option_results[candidate]["combined"]
                )
                - second_score
                >= 0.08
            ):
                answer = candidate
                status = "answered"

        elif len(probable) >= 2:
            top_two = probable[:2]

            if all(
                option_results[option]["ml_filled"] >= 0.55
                for option in top_two
            ):
                answer = "MULTIPLE"
                status = "multiple"
                multiple_options = top_two

        if answer is None:
            blank_count = sum(
                bool(
                    option_results[option]["verified_blank"]
                )
                for option in option_results
            )

            top_score = float(
                option_results[ranked[0]]["combined"]
            )

            if (
                blank_count >= len(option_results) - 1
                or top_score < 0.34
            ):
                status = "blank"

            elif (
                old_answer in options
                and float(
                    option_results[old_answer]["combined"]
                )
                >= 0.46
                and not option_results[old_answer]["verified_blank"]
            ):
                answer = old_answer
                status = "answered"

            elif old_answer == "MULTIPLE":
                old_supported = [
                    option
                    for option in ranked
                    if (
                        option_results[option]["combined"] >= 0.48
                        and (
                            float(
                                option_results[option]["reference"].get(
                                    "score",
                                    0.0,
                                )
                            )
                            >= 0.15
                            or float(
                                option_results[option]["solid"].get(
                                    "score",
                                    0.0,
                                )
                            )
                            >= 0.52
                        )
                    )
                ]

                if len(old_supported) >= 2:
                    answer = "MULTIPLE"
                    status = "multiple"
                    multiple_options = old_supported[:2]
                elif len(old_supported) == 1:
                    answer = old_supported[0]
                    status = "answered"

        final_answers[question_key] = answer

        updated_debug = dict(existing_debug)
        updated_debug["answer"] = answer
        updated_debug["status"] = status
        updated_debug["multiple_options"] = multiple_options
        updated_debug["ml_consensus_profile"] = PROFILE
        updated_debug["ml_consensus_primary"] = True
        updated_debug["ml_consensus_old_answer"] = old_answer
        updated_debug["ml_consensus_options"] = option_results
        updated_debug["ml_consensus_ranked"] = ranked

        final_debug[question_key] = updated_debug

    return final_answers, final_debug


def rescue_jee_multiple_from_ml_debug(
    *,
    gray,
    ml_answer,
    ml_decision,
):
    """
    JEE final solid-center double-mark rescue.

    Runs after hybrid ML, so a row can still recover MULTIPLE even when the
    first classifier returned BLANK, AMBIGUOUS, or a single option.
    """
    if not isinstance(ml_decision, dict):
        return ml_answer, ml_decision

    option_data = ml_decision.get("options", {})

    if not isinstance(option_data, dict):
        return ml_answer, ml_decision

    options = [
        option
        for option in ("A", "B", "C", "D")
        if option in option_data
    ]

    if len(options) != 4:
        return ml_answer, ml_decision

    evidence = {}

    for option in options:
        center = option_data[option].get("crop_center")

        if (
            not isinstance(center, (list, tuple))
            or len(center) < 2
        ):
            return ml_answer, ml_decision

        solid = _solid_center_evidence(
            gray,
            float(center[0]),
            float(center[1]),
            search_radius=4,
        )

        ml_filled = _ml_probability(
            option_data[option],
            "filled",
        )

        solid_score = float(
            solid.get("score", 0.0)
        )

        verified = (
            (
                solid_score >= 0.64
                and ml_filled >= 0.25
            )
            or solid_score >= 0.76
        )

        evidence[option] = {
            "solid": solid,
            "ml_filled": round(ml_filled, 4),
            "verified": bool(verified),
            "combined": round(
                0.74 * solid_score
                + 0.26 * ml_filled,
                4,
            ),
        }

    ranked = sorted(
        options,
        key=lambda option:
            float(evidence[option]["combined"]),
        reverse=True,
    )

    verified = [
        option
        for option in ranked
        if evidence[option]["verified"]
    ]

    if len(verified) < 2:
        return ml_answer, ml_decision

    non_verified = [
        option
        for option in ranked
        if option not in verified
    ]

    second_score = float(
        evidence[verified[1]]["combined"]
    )

    best_blank_score = (
        max(
            float(evidence[option]["combined"])
            for option in non_verified
        )
        if non_verified
        else 0.0
    )

    if second_score - best_blank_score < 0.10:
        return ml_answer, ml_decision

    rescued = dict(ml_decision)
    rescued["answer"] = "MULTIPLE"
    rescued["status"] = "multiple"
    rescued["multiple_options"] = verified
    rescued["best_option"] = verified[0]
    rescued["jee_solid_multiple_rescue"] = True
    rescued["jee_solid_multiple_profile"] = PROFILE
    rescued["jee_solid_multiple_evidence"] = evidence

    return "MULTIPLE", rescued
