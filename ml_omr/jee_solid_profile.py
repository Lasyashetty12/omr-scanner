from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


PROFILE = "jee_solid_profile_ml_v10_34"
OPTIONS = ("A", "B", "C", "D")


def _gray(image):
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _core_p75(gray: np.ndarray, center, radius: int = 6) -> float:
    x = int(round(float(center[0])))
    y = int(round(float(center[1])))
    radius = max(3, int(radius))

    h, w = gray.shape[:2]
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 255.0

    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = ((xx - x) ** 2 + (yy - y) ** 2) <= radius ** 2
    pixels = roi[mask].astype(np.float32)

    if pixels.size == 0:
        return 255.0

    # Broad-ink statistic:
    # a printed A/B/C/D glyph can darken the mean/core, but usually cannot
    # lower P75 unless a substantial part of the bubble interior is shaded.
    return float(np.percentile(pixels, 75.0))


def _ml_prob(info: Dict[str, Any], label: str) -> float:
    direct = f"ml_{label}_probability"

    if direct in info:
        try:
            return float(info.get(direct, 0.0))
        except (TypeError, ValueError):
            return 0.0

    ml = info.get("ml", {})
    if not isinstance(ml, dict):
        return 0.0

    probabilities = ml.get("probabilities", {})
    if isinstance(probabilities, dict):
        try:
            return float(probabilities.get(label, 0.0))
        except (TypeError, ValueError):
            return 0.0

    return 0.0


def _record(mapping, question):
    if not isinstance(mapping, dict):
        return {}

    result = mapping.get(
        question,
        mapping.get(str(question), {}),
    )

    return result if isinstance(result, dict) else {}


def _answer(record) -> str:
    if not isinstance(record, dict):
        return ""
    return str(record.get("answer", "") or "").strip().upper()


def _cluster_from_values(
    values: Dict[str, float],
    column_darkening: Dict[str, float],
    *,
    split_gap: float = 12.0,
    minimum_column_darkening: float = 9.0,
):
    """
    Find a physically separated dark group in one A/B/C/D row.

    True marks form a low-P75 cluster separated from blank bubbles.
    Uniform lighting changes or a single thick printed glyph do not.
    """
    ranked = sorted(
        values.items(),
        key=lambda item: float(item[1]),
    )

    if len(ranked) != 4:
        return {
            "candidate_options": [],
            "max_gap": 0.0,
            "spread": 0.0,
            "ranked": ranked,
            "gaps": [],
        }

    gaps = [
        float(ranked[i + 1][1] - ranked[i][1])
        for i in range(3)
    ]

    max_gap = max(gaps)
    split_index = int(np.argmax(np.asarray(gaps, dtype=np.float32)))
    spread = float(ranked[-1][1] - ranked[0][1])

    candidates = []

    if max_gap >= float(split_gap):
        proposed = [
            option
            for option, _ in ranked[: split_index + 1]
        ]

        if 1 <= len(proposed) <= 3:
            if all(
                float(column_darkening.get(option, 0.0))
                >= float(minimum_column_darkening)
                for option in proposed
            ):
                candidates = proposed

    return {
        "candidate_options": candidates,
        "max_gap": round(max_gap, 3),
        "spread": round(spread, 3),
        "ranked": [
            [option, round(float(value), 3)]
            for option, value in ranked
        ],
        "gaps": [round(float(gap), 3) for gap in gaps],
    }


def apply_jee_solid_profile_overrides(
    *,
    gray,
    coordinates,
    stable_mcq,
    ml_answers,
    ml_debug,
    template,
):
    """
    JEE-MCQ-only correction layer.

    It keeps the v10.33 reference reader and the normal ml_omr pass, but fixes
    a specific failure mode seen on dim/blurred mobile captures:

      * false SINGLE from a dark printed glyph / local reference mismatch
      * missed light broad fill
      * missed second broad fill

    No question numbers or answer key values are hard-coded.
    KCET/NEET never call this function.
    """
    gray = _gray(gray)

    core_radius = int(
        template.get("jee_solid_profile_core_radius", 6)
    )
    split_gap = float(
        template.get("jee_solid_profile_split_gap", 12.0)
    )
    minimum_column_darkening = float(
        template.get("jee_solid_profile_column_darkening", 9.0)
    )

    stable_out = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in (stable_mcq or {}).items()
    }
    ml_answers_out = dict(ml_answers or {})
    ml_debug_out = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in (ml_debug or {}).items()
    }

    p75 = {}

    for question, option_map in (coordinates or {}).items():
        try:
            q = int(question)
        except (TypeError, ValueError):
            continue

        if not isinstance(option_map, dict):
            continue

        row = {}
        for option in OPTIONS:
            center = option_map.get(option)
            if (
                not isinstance(center, (list, tuple))
                or len(center) < 2
            ):
                continue

            row[option] = _core_p75(
                gray,
                center,
                radius=core_radius,
            )

        if len(row) == 4:
            p75[q] = row

    for section in template.get("mcq_sections", []):
        start = int(section["start_question"])
        total = int(section["total_questions"])
        questions = list(range(start, start + total))

        column_baseline = {}

        for option in OPTIONS:
            samples = [
                float(p75[q][option])
                for q in questions
                if q in p75
            ]

            if samples:
                # Brighter end of the distribution approximates blank tone.
                column_baseline[option] = float(
                    np.percentile(
                        np.asarray(samples, dtype=np.float32),
                        75.0,
                    )
                )

        for question in questions:
            if question not in p75 or len(column_baseline) != 4:
                continue

            values = p75[question]

            darkening = {
                option: (
                    float(column_baseline[option])
                    - float(values[option])
                )
                for option in OPTIONS
            }

            cluster = _cluster_from_values(
                values,
                darkening,
                split_gap=split_gap,
                minimum_column_darkening=minimum_column_darkening,
            )

            stable_record = _record(stable_out, question)
            stable_answer = _answer(stable_record)
            ml_decision = _record(ml_debug_out, question)

            option_data = ml_decision.get("options", {})
            if not isinstance(option_data, dict):
                option_data = {}

            robust_scores = stable_record.get("scores", {})
            if not isinstance(robust_scores, dict):
                robust_scores = {}

            candidates = list(cluster["candidate_options"])
            accepted = []

            for option in candidates:
                info = option_data.get(option, {})
                if not isinstance(info, dict):
                    info = {}

                ml_filled = _ml_prob(info, "filled")
                ml_blank = _ml_prob(info, "blank")

                try:
                    robust_score = float(
                        robust_scores.get(option, 0.0)
                    )
                except (TypeError, ValueError):
                    robust_score = 0.0

                # ML is a veto/support signal, not a source of extra options.
                decisive_ml_blank = (
                    ml_blank >= 0.78
                    and ml_filled <= 0.22
                )

                supported = (
                    not decisive_ml_blank
                    and (
                        ml_filled >= 0.35
                        or robust_score >= 0.080
                        or float(darkening[option]) >= 18.0
                    )
                )

                if supported:
                    accepted.append(option)

            override = None

            # One physically isolated broad mark.
            if len(accepted) == 1:
                if float(cluster["max_gap"]) >= max(split_gap, 14.0):
                    override = accepted[0]

            # Two or three physically isolated broad marks -> invalid MULTIPLE.
            elif len(accepted) >= 2:
                override = "MULTIPLE"

            # No broad-fill group: allow a false relaxed SINGLE to become BLANK.
            elif stable_answer in OPTIONS:
                max_gap = float(cluster["max_gap"])
                spread = float(cluster["spread"])

                stable_info = option_data.get(stable_answer, {})
                if not isinstance(stable_info, dict):
                    stable_info = {}

                ml_filled = _ml_prob(stable_info, "filled")
                ml_blank = _ml_prob(stable_info, "blank")

                try:
                    robust_score = float(
                        robust_scores.get(
                            stable_answer,
                            stable_record.get("highest_score", 0.0),
                        )
                    )
                except (TypeError, ValueError):
                    robust_score = 0.0

                physical_blank = (
                    max_gap < split_gap
                    and spread < 22.0
                )

                # 0.145 is the JEE reference reader's normal filled threshold.
                # Values below ~0.155 are the borderline/relaxed region.
                weak_reference_single = robust_score < 0.155

                ml_not_strong_fill = (
                    ml_filled < 0.72
                    or ml_blank >= ml_filled
                )

                if (
                    physical_blank
                    and weak_reference_single
                    and ml_not_strong_fill
                ):
                    override = "BLANK"

            debug = dict(ml_decision)
            debug["jee_solid_profile"] = {
                "profile": PROFILE,
                "p75": {
                    option: round(float(values[option]), 3)
                    for option in OPTIONS
                },
                "column_darkening": {
                    option: round(float(darkening[option]), 3)
                    for option in OPTIONS
                },
                "cluster": cluster,
                "accepted_options": list(accepted),
                "override": override,
            }

            if override == "MULTIPLE":
                stable_new = dict(stable_record)
                stable_new["answer"] = "MULTIPLE"
                stable_new["multiple_options"] = list(accepted)
                stable_new["reader"] = PROFILE
                stable_out[question] = stable_new

                ml_answers_out[question] = "MULTIPLE"
                debug["answer"] = "MULTIPLE"
                debug["status"] = "multiple"
                debug["multiple_options"] = list(accepted)
                debug["best_option"] = accepted[0]

            elif override in OPTIONS:
                stable_new = dict(stable_record)
                stable_new["answer"] = override
                stable_new["multiple_options"] = []
                stable_new["reader"] = PROFILE
                stable_out[question] = stable_new

                ml_answers_out[question] = override
                debug["answer"] = override
                debug["status"] = "answered"
                debug["multiple_options"] = []
                debug["best_option"] = override

            elif override == "BLANK":
                stable_new = dict(stable_record)
                stable_new["answer"] = "BLANK"
                stable_new["multiple_options"] = []
                stable_new["reader"] = PROFILE
                stable_out[question] = stable_new

                ml_answers_out[question] = None
                debug["answer"] = None
                debug["status"] = "blank"
                debug["multiple_options"] = []

            ml_debug_out[question] = debug

    return stable_out, ml_answers_out, ml_debug_out
