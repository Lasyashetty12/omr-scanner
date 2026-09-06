from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


PROFILE = "conservative_final_guard_v10_29"


def _ml_probability(option_data: Dict[str, Any], label: str) -> float:
    direct_key = f"ml_{label}_probability"

    if direct_key in option_data:
        try:
            return float(option_data.get(direct_key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    prediction = option_data.get("ml", {})

    if not isinstance(prediction, dict):
        return 0.0

    probabilities = prediction.get("probabilities", {})

    if isinstance(probabilities, dict):
        try:
            return float(probabilities.get(label, 0.0))
        except (TypeError, ValueError):
            return 0.0

    predicted_label = str(
        prediction.get("label", "")
    ).strip().lower()

    try:
        confidence = float(
            prediction.get("confidence", 0.0)
        )
    except (TypeError, ValueError):
        confidence = 0.0

    return confidence if predicted_label == label else 0.0


def apply_strict_ml_blank_veto(
    *,
    decision: Dict[str, Any],
    option_data: Dict[str, Dict[str, Any]],
    questions_per_column: int,
) -> Dict[str, Any]:
    """
    Conservative NEET/KCET-only final guard.

    It NEVER promotes BLANK/AMBIGUOUS to a filled answer.
    It only removes an already-selected option when the existing ONNX model
    is overwhelmingly confident that the selected physical bubble is blank.

    This preserves the older, better-performing reader while fixing the
    specific false-filled failure class.
    """
    if int(questions_per_column) >= 1000:
        return decision

    if not isinstance(decision, dict):
        return decision

    status = str(
        decision.get("status", "")
    ).strip().lower()

    options = [
        option
        for option in ("A", "B", "C", "D")
        if option in option_data
    ]

    if len(options) < 2:
        return decision

    def is_decisive_blank(option: str) -> bool:
        info = option_data.get(option, {})

        blank = _ml_probability(info, "blank")
        filled = _ml_probability(info, "filled")
        ambiguous = _ml_probability(info, "ambiguous")

        metrics = info.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        try:
            disk = float(
                metrics.get("disk_dark_ratio", 0.0)
            )
        except (TypeError, ValueError):
            disk = 0.0

        try:
            darkness = float(
                metrics.get("center_darkness", 0.0)
            )
        except (TypeError, ValueError):
            darkness = 0.0

        # Very high model confidence is mandatory. The image limits are only
        # safety guards so a truly dark handwritten fill is not blanked merely
        # because the model had one bad prediction.
        return bool(
            blank >= 0.94
            and filled <= 0.045
            and ambiguous <= 0.12
            and disk <= 0.66
            and darkness <= 105.0
        )

    if status == "answered":
        answer = str(
            decision.get("answer", "")
        ).strip().upper()

        if (
            answer in option_data
            and is_decisive_blank(answer)
        ):
            corrected = dict(decision)
            corrected["answer"] = None
            corrected["status"] = "blank"
            corrected["multiple_options"] = []
            corrected["strict_ml_blank_veto"] = True
            corrected["strict_ml_blank_veto_option"] = answer
            corrected["strict_ml_blank_veto_profile"] = PROFILE
            return corrected

        return decision

    if status == "multiple":
        multiple_options = [
            str(option).upper()
            for option in (
                decision.get("multiple_options", [])
                or []
            )
            if str(option).upper() in option_data
        ]

        if not multiple_options:
            return decision

        kept = [
            option
            for option in multiple_options
            if not is_decisive_blank(option)
        ]

        if len(kept) == len(multiple_options):
            return decision

        corrected = dict(decision)
        corrected["strict_ml_blank_veto"] = True
        corrected["strict_ml_blank_veto_profile"] = PROFILE
        corrected["strict_ml_removed_options"] = [
            option
            for option in multiple_options
            if option not in kept
        ]

        if len(kept) >= 2:
            corrected["answer"] = "MULTIPLE"
            corrected["status"] = "multiple"
            corrected["multiple_options"] = kept
            corrected["best_option"] = kept[0]
            return corrected

        if len(kept) == 1:
            corrected["answer"] = kept[0]
            corrected["status"] = "answered"
            corrected["multiple_options"] = []
            corrected["best_option"] = kept[0]
            return corrected

        corrected["answer"] = None
        corrected["status"] = "blank"
        corrected["multiple_options"] = []
        return corrected

    return decision


def resolve_strict_jee_secondary_multiple(
    *,
    stable_answer: Any,
    ml_answer: Any,
    ml_decision: Dict[str, Any],
):
    """
    Narrow JEE second-mark rescue.

    Unlike the broad v10.26/v10.28 rescues, this runs ONLY when the proven
    stable reader already has one concrete A/B/C/D answer. A second option is
    added only if exactly one other bubble has strong ONNX + physical-fill
    evidence. It never turns a blank/uncertain JEE row into MULTIPLE.
    """
    stable = str(
        stable_answer or ""
    ).strip().upper()

    if stable not in ("A", "B", "C", "D"):
        return ml_answer, ml_decision

    if not isinstance(ml_decision, dict):
        return ml_answer, ml_decision

    option_data = ml_decision.get("options", {})

    if not isinstance(option_data, dict):
        return ml_answer, ml_decision

    if stable not in option_data:
        return ml_answer, ml_decision

    def evidence(option: str):
        info = option_data.get(option, {})
        metrics = info.get("metrics", {})

        if not isinstance(metrics, dict):
            metrics = {}

        filled = _ml_probability(info, "filled")
        blank = _ml_probability(info, "blank")

        def metric(name, default=0.0):
            try:
                return float(
                    metrics.get(name, default)
                )
            except (TypeError, ValueError):
                return float(default)

        return {
            "filled": filled,
            "blank": blank,
            "disk": metric("disk_dark_ratio"),
            "core": metric("core_dark_ratio"),
            "darkness": metric("center_darkness"),
        }

    primary = evidence(stable)

    primary_supported = bool(
        (
            primary["filled"] >= 0.55
            and primary["blank"] <= 0.35
        )
        or (
            primary["disk"] >= 0.72
            and primary["darkness"] >= 82.0
        )
    )

    if not primary_supported:
        return ml_answer, ml_decision

    candidates = []

    for option in ("A", "B", "C", "D"):
        if option == stable or option not in option_data:
            continue

        data = evidence(option)

        strong_model_path = bool(
            data["filled"] >= 0.72
            and data["blank"] <= 0.20
            and data["disk"] >= 0.50
            and data["darkness"] >= 65.0
        )

        strong_visual_path = bool(
            data["filled"] >= 0.60
            and data["blank"] <= 0.28
            and data["disk"] >= 0.72
            and data["core"] >= 0.70
            and data["darkness"] >= 82.0
        )

        if strong_model_path or strong_visual_path:
            candidates.append(
                (
                    data["filled"],
                    data["disk"],
                    data["darkness"],
                    option,
                    data,
                )
            )

    # Exactly one secondary fill is required. If two or three blank printed
    # rings look suspicious, preserve the old stable single instead.
    if len(candidates) != 1:
        return ml_answer, ml_decision

    candidates.sort(reverse=True)
    _, _, _, second_option, second_evidence = candidates[0]

    rescued = dict(ml_decision)
    rescued["answer"] = "MULTIPLE"
    rescued["status"] = "multiple"
    rescued["best_option"] = stable
    rescued["multiple_options"] = [
        stable,
        second_option,
    ]
    rescued["strict_jee_secondary_multiple"] = True
    rescued["strict_jee_secondary_option"] = second_option
    rescued["strict_jee_secondary_evidence"] = second_evidence
    rescued["strict_jee_secondary_profile"] = PROFILE

    return "MULTIPLE", rescued


def _series_local_score(
    gray: np.ndarray,
    center_x: float,
    center_y: float,
    search_radius: int,
):
    if gray.ndim == 3:
        gray = cv2.cvtColor(
            gray,
            cv2.COLOR_BGR2GRAY,
        )

    radius = 14
    height, width = gray.shape[:2]

    yy, xx = np.ogrid[
        -radius:radius + 1,
        -radius:radius + 1,
    ]

    distance_sq = xx * xx + yy * yy
    core_mask = distance_sq <= 5 * 5
    disk_mask = distance_sq <= 8 * 8
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
            ].astype(np.float32)

            core = patch[core_mask]
            disk = patch[disk_mask]
            paper = patch[paper_mask]

            if (
                core.size < 20
                or disk.size < 70
                or paper.size < 80
            ):
                continue

            paper_level = float(
                np.percentile(paper, 75.0)
            )

            dark_threshold = float(
                np.clip(
                    paper_level - 38.0,
                    70.0,
                    190.0,
                )
            )

            core_mean = float(np.mean(core))
            disk_mean = float(np.mean(disk))
            core_delta = max(
                0.0,
                paper_level - core_mean,
            )
            disk_delta = max(
                0.0,
                paper_level - disk_mean,
            )

            core_ratio = float(
                np.mean(core < dark_threshold)
            )
            disk_ratio = float(
                np.mean(disk < dark_threshold)
            )

            score = (
                0.38
                * float(
                    np.clip(
                        core_delta / 105.0,
                        0.0,
                        1.0,
                    )
                )
                + 0.22
                * float(
                    np.clip(
                        disk_delta / 90.0,
                        0.0,
                        1.0,
                    )
                )
                + 0.24 * core_ratio
                + 0.16 * disk_ratio
                - 0.004
                * (
                    abs(dx)
                    + abs(dy)
                )
            )

            record = {
                "score": float(score),
                "center": [cx, cy],
                "core_delta": core_delta,
                "disk_delta": disk_delta,
                "core_ratio": core_ratio,
                "disk_ratio": disk_ratio,
            }

            if (
                best is None
                or record["score"] > best["score"]
            ):
                best = record

    return best


def detect_series_cv_fallback(
    gray_image: np.ndarray,
    template: Dict[str, Any],
    exam_name: str | None = None,
):
    """
    Conservative P/Q/R/S fallback used only when the existing detector fails.

    It searches a small local area around each configured series bubble and
    chooses the option with the strongest SOLID central ink, rather than the
    printed ring. It cannot jump between P/Q/R/S because their centres are far
    apart relative to the search window.
    """
    series_config = template.get("series", {})

    if not isinstance(series_config, dict):
        return None

    coordinates = series_config.get(
        "coordinates",
        {},
    )

    if not isinstance(coordinates, dict) or not coordinates:
        return None

    search_radius = int(
        series_config.get(
            "cv_fallback_search_radius",
            12,
        )
    )

    search_radius = max(
        5,
        min(
            15,
            search_radius,
        ),
    )

    details = {}

    for label, point in coordinates.items():
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
        ):
            continue

        result = _series_local_score(
            gray_image,
            float(point[0]),
            float(point[1]),
            search_radius,
        )

        if result is not None:
            details[str(label)] = result

    if len(details) < 2:
        return None

    ranked = sorted(
        details,
        key=lambda label:
            float(details[label]["score"]),
        reverse=True,
    )

    best_label = ranked[0]
    second_label = ranked[1]

    best = details[best_label]
    second = details[second_label]

    best_score = float(best["score"])
    gap = (
        best_score
        - float(second["score"])
    )

    # Require both a meaningful absolute solid-fill score and a lead over the
    # next series bubble. This avoids guessing on an unfilled/ambiguous row.
    accepted = bool(
        (
            best_score >= 0.50
            and gap >= 0.075
            and best["core_ratio"] >= 0.52
        )
        or (
            best_score >= 0.64
            and gap >= 0.040
            and best["core_ratio"] >= 0.68
        )
    )

    if not accepted:
        return None

    return {
        "value": best_label,
        "score": round(best_score, 4),
        "confidence_gap": round(gap, 4),
        "scores": {
            label: round(
                float(details[label]["score"]),
                4,
            )
            for label in details
        },
        "sampling_centres": {
            label: details[label]["center"]
            for label in details
        },
        "reader": "cv_series_solid_core_fallback_v10_29",
        "exam": str(
            exam_name
            or template.get("exam_name")
            or ""
        ).strip().upper(),
    }
