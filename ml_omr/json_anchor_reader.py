
from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from ml_omr.hybrid_reader import scan_answers_ml
from ml_omr.inference import classify_batch


VALID_OPTIONS = {"A", "B", "C", "D"}


def build_template_json_coordinates(template: Dict[str, Any]):
    """Build every answer bubble position directly from the template JSON."""
    options = [str(v) for v in template.get("options", ["A", "B", "C", "D"])]
    columns = template.get("columns") or []
    y_positions = template.get("question_y_positions") or []
    qpc = int(template.get("questions_per_column", len(y_positions)))
    total = int(template.get("total_questions", len(columns) * qpc))

    if not columns or len(y_positions) < qpc:
        raise ValueError("Template JSON answer coordinates are incomplete.")

    coordinates = {}

    for col_index, column in enumerate(columns):
        for row_index in range(qpc):
            question = col_index * qpc + row_index + 1
            if question > total:
                break

            y = float(y_positions[row_index])
            coordinates[question] = {
                option: (float(column[option]), y)
                for option in options
            }

    return coordinates


def _stabilize_json_coordinates(
    json_coordinates,
    fitted_coordinates,
    template,
    max_shift=6.0,
):
    """
    JSON keeps the exact row/option layout. The detected grid may contribute
    only a small robust median translation per 60-question column.
    """
    fitted_coordinates = fitted_coordinates or {}
    qpc = int(template.get("questions_per_column", 60))
    col_count = len(template.get("columns") or [])

    samples = {
        index: {"dx": [], "dy": []}
        for index in range(col_count)
    }

    for question, option_map in json_coordinates.items():
        col_index = (int(question) - 1) // qpc
        fitted_map = fitted_coordinates.get(question) or {}

        if col_index not in samples:
            continue

        for option, json_point in option_map.items():
            fitted_point = fitted_map.get(option)

            if not fitted_point or len(fitted_point) < 2:
                continue

            try:
                dx = float(fitted_point[0]) - float(json_point[0])
                dy = float(fitted_point[1]) - float(json_point[1])
            except (TypeError, ValueError):
                continue

            # Reject a bad fitted-grid jump instead of moving JSON geometry.
            if abs(dx) <= 18.0:
                samples[col_index]["dx"].append(dx)
            if abs(dy) <= 18.0:
                samples[col_index]["dy"].append(dy)

    offsets = {}

    for col_index in range(col_count):
        dxs = samples[col_index]["dx"]
        dys = samples[col_index]["dy"]

        dx = float(np.median(dxs)) if dxs else 0.0
        dy = float(np.median(dys)) if dys else 0.0

        dx = float(np.clip(dx, -max_shift, max_shift))
        dy = float(np.clip(dy, -max_shift, max_shift))

        offsets[col_index] = {
            "dx": dx,
            "dy": dy,
            "samples_x": len(dxs),
            "samples_y": len(dys),
        }

    stabilized = {}

    for question, option_map in json_coordinates.items():
        col_index = (int(question) - 1) // qpc
        offset = offsets.get(col_index, {"dx": 0.0, "dy": 0.0})

        stabilized[question] = {
            option: (
                float(point[0]) + float(offset["dx"]),
                float(point[1]) + float(offset["dy"]),
            )
            for option, point in option_map.items()
        }

    return stabilized, offsets


def _is_single(value):
    return isinstance(value, str) and value.strip().upper() in VALID_OPTIONS


def _strength(details):
    details = details or {}

    if str(details.get("status", "")).lower() != "answered":
        return 0.0

    disk_gap = max(0.0, float(details.get("disk_gap", 0.0) or 0.0))
    top_gap = max(0.0, float(details.get("top_gap", 0.0) or 0.0))
    darkness = max(0.0, float(details.get("best_darkness", 0.0) or 0.0))

    return (
        min(disk_gap / 0.16, 1.0)
        + min(top_gap / 35.0, 1.0)
        + 0.35 * min(darkness / 90.0, 1.0)
    )


def scan_answers_json_anchored(
    gray,
    template,
    fitted_coordinates,
    crop_radius=12,
):
    """
    KCET/NEET:
      1. JSON positions are authoritative.
      2. Only a tiny per-column translation is allowed.
      3. Existing ONNX/ML reader classifies those exact bubbles.
      4. Existing fitted-grid ML can only rescue a strong JSON blank.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    json_coordinates = build_template_json_coordinates(template)
    anchored_coordinates, offsets = _stabilize_json_coordinates(
        json_coordinates,
        fitted_coordinates,
        template,
    )

    json_answers, json_debug = scan_answers_ml(
        gray=gray,
        coordinates=anchored_coordinates,
        crop_radius=int(crop_radius),
            questions_per_column=int(template.get("questions_per_column", 60)),
)

    fitted_answers, fitted_debug = scan_answers_ml(
        gray=gray,
        coordinates=fitted_coordinates,
        crop_radius=int(crop_radius),
            questions_per_column=int(template.get("questions_per_column", 60)),
)

    final_answers = {}
    final_debug = {}

    for question in json_coordinates:
        json_answer = json_answers.get(question)
        fitted_answer = fitted_answers.get(question)
        jd = json_debug.get(question, {}) or {}
        fd = fitted_debug.get(question, {}) or {}

        source = "json_ml"
        selected = jd

        if str(json_answer).upper() == "MULTIPLE":
            final = "MULTIPLE"
        elif _is_single(json_answer):
            final = str(json_answer).upper()
        elif _is_single(fitted_answer) and _strength(fd) >= 1.10:
            final = str(fitted_answer).upper()
            source = "fitted_ml_rescue"
            selected = fd
        else:
            final = None

        final_answers[question] = final
        final_debug[question] = {
            **selected,
            "answer": final,
            "geometry_source": source,
            "json_answer": json_answer,
            "fitted_answer": fitted_answer,
            "json_strength": round(_strength(jd), 4),
            "fitted_strength": round(_strength(fd), 4),
        }

    final_debug["_json_anchor"] = {
        "reader": "kcet_neet_json_ml_v10_19",
        "template_is_geometry_authority": True,
        "column_offsets": {
            str(index + 1): {
                "dx": round(float(values["dx"]), 3),
                "dy": round(float(values["dy"]), 3),
                "samples_x": int(values["samples_x"]),
                "samples_y": int(values["samples_y"]),
            }
            for index, values in offsets.items()
        },
    }

    return final_answers, final_debug


def _crop(gray, x, y, radius=12):
    x = int(round(float(x)))
    y = int(round(float(y)))
    h, w = gray.shape[:2]

    return gray[
        max(0, y - radius):min(h, y + radius + 1),
        max(0, x - radius):min(w, x + radius + 1),
    ]


def _disk_ratio(gray, x, y, radius=8):
    x = int(round(float(x)))
    y = int(round(float(y)))
    h, w = gray.shape[:2]

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    roi = gray[y0:y1, x0:x1]

    if roi.size == 0:
        return 0.0

    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = ((xx - x) ** 2 + (yy - y) ** 2) <= radius ** 2
    pixels = roi[mask].astype(np.float32)

    if pixels.size == 0:
        return 0.0

    paper = float(np.percentile(roi, 82.0))
    threshold = float(np.clip(paper - 34.0, 75.0, 175.0))

    return float(np.mean(pixels < threshold))


def _read_choice_group_ml(gray, config):
    """
    Read Class / Exam bubbles from their JSON coordinates.
    Small +/-6 px search handles the remaining camera alignment error.
    """
    choices = config.get("choices") or {}

    if not choices:
        return {"value": None, "reader": "json_choice_ml_v10_19"}

    offsets = (-6, -3, 0, 3, 6)
    crops = []
    meta = []

    for label, point in choices.items():
        bx, by = float(point[0]), float(point[1])

        for dy in offsets:
            for dx in offsets:
                x, y = bx + dx, by + dy
                crops.append(_crop(gray, x, y, 12))
                meta.append((str(label), x, y))

    predictions = classify_batch(crops)
    best_by_label = {}

    for (label, x, y), prediction in zip(meta, predictions):
        probs = prediction.get("probabilities", {}) if isinstance(prediction, dict) else {}
        ml_filled = float(probs.get("filled", 0.0))
        ml_blank = float(probs.get("blank", 0.0))
        disk = _disk_ratio(gray, x, y)
        score = 0.78 * ml_filled + 0.22 * disk

        current = best_by_label.get(label)

        if current is None or score > current["score"]:
            best_by_label[label] = {
                "score": score,
                "ml_filled": ml_filled,
                "ml_blank": ml_blank,
                "disk_ratio": disk,
                "center": [round(x, 2), round(y, 2)],
            }

    ranked = sorted(
        best_by_label.items(),
        key=lambda item: item[1]["score"],
        reverse=True,
    )

    if not ranked:
        return {"value": None, "reader": "json_choice_ml_v10_19"}

    best_label, best = ranked[0]
    second_score = ranked[1][1]["score"] if len(ranked) > 1 else 0.0
    gap = float(best["score"] - second_score)

    strong = (
        (
            float(best["ml_filled"]) >= 0.44
            or float(best["disk_ratio"]) >= 0.43
        )
        and gap >= 0.055
    )

    return {
        "value": best_label if strong else None,
        "reader": "json_choice_ml_v10_19",
        "best_label": best_label,
        "best_score": round(float(best["score"]), 4),
        "confidence_gap": round(gap, 4),
        "scores": {
            label: {
                "score": round(float(values["score"]), 4),
                "ml_filled": round(float(values["ml_filled"]), 4),
                "ml_blank": round(float(values["ml_blank"]), 4),
                "disk_ratio": round(float(values["disk_ratio"]), 4),
                "center": values["center"],
            }
            for label, values in best_by_label.items()
        },
    }


def recover_identity_choices_ml(image, template):
    """
    Recover Class and KCET/NEET Exam using template JSON + the ONNX bubble ML.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    identity_config = template.get("identity") or {}
    result = {}

    for field in ("class", "exam"):
        config = identity_config.get(field) or {}

        if not config:
            continue

        details = _read_choice_group_ml(gray, config)
        result[field] = details.get("value")
        result[f"{field}_details"] = details

    result["reader"] = "kcet_neet_identity_json_ml_v10_19"
    return result
