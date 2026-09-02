
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from pathlib import Path

import cv2
import numpy as np


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _core_fill_ratio(
    gray: np.ndarray,
    x: float,
    y: float,
    *,
    radius: int = 6,
    dark_threshold: int = 140,
) -> float:
    x = int(round(float(x)))
    y = int(round(float(y)))
    radius = max(2, int(radius))
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
    count = int(np.count_nonzero(mask))
    if count <= 0:
        return 0.0

    return float(np.count_nonzero((roi < dark_threshold) & mask)) / float(count)


_JEE_REFERENCE_GRAY_CACHE: Dict[str, np.ndarray] = {}


def _get_jee_reference_gray(
    template: Dict[str, Any],
) -> np.ndarray | None:
    reference_name = str(
        template.get(
            "reference_image",
            "jee_generated.png",
        )
    )

    if reference_name in _JEE_REFERENCE_GRAY_CACHE:
        return _JEE_REFERENCE_GRAY_CACHE[
            reference_name
        ]

    path = (
        Path(__file__).resolve().parent
        / "references"
        / reference_name
    )

    if not path.exists():
        return None

    reference = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if reference is None:
        return None

    _JEE_REFERENCE_GRAY_CACHE[
        reference_name
    ] = reference

    return reference


def _extract_square(
    gray: np.ndarray,
    x: float,
    y: float,
    radius: int,
) -> np.ndarray | None:
    x = int(round(float(x)))
    y = int(round(float(y)))
    radius = int(radius)

    x0 = x - radius
    x1 = x + radius + 1
    y0 = y - radius
    y1 = y + radius + 1

    if (
        x0 < 0
        or y0 < 0
        or x1 > gray.shape[1]
        or y1 > gray.shape[0]
    ):
        return None

    return gray[
        y0:y1,
        x0:x1,
    ].astype(
        np.float32
    )


def _extra_ink_score(
    gray: np.ndarray,
    reference_gray: np.ndarray | None,
    actual_x: float,
    actual_y: float,
    reference_x: float,
    reference_y: float,
    *,
    core_radius: int = 5,
    outer_radius: int = 9,
    reference_search: int = 2,
    actual_search: int = 2,
) -> float:
    if reference_gray is None:
        return _core_fill_ratio(
            gray,
            actual_x,
            actual_y,
            radius=core_radius,
            dark_threshold=140,
        )

    size = int(outer_radius) * 2 + 1

    yy, xx = np.ogrid[
        :size,
        :size,
    ]

    centre = int(outer_radius)

    distance_sq = (
        (xx - centre) ** 2
        + (yy - centre) ** 2
    )

    core_mask = (
        distance_sq
        <= int(core_radius) ** 2
    )

    annulus_mask = (
        (
            distance_sq
            >= (int(core_radius) + 2) ** 2
        )
        & (
            distance_sq
            <= int(outer_radius) ** 2
        )
    )

    best_reference = None
    best_current = None
    best_annulus_error = None

    for current_dx in range(
        -int(actual_search),
        int(actual_search) + 1,
    ):
        for current_dy in range(
            -int(actual_search),
            int(actual_search) + 1,
        ):
            current = _extract_square(
                gray,
                float(actual_x) + current_dx,
                float(actual_y) + current_dy,
                int(outer_radius),
            )

            if current is None:
                continue

            current_annulus = current[
                annulus_mask
            ]

            if current_annulus.size == 0:
                continue

            for reference_dx in range(
                -int(reference_search),
                int(reference_search) + 1,
            ):
                for reference_dy in range(
                    -int(reference_search),
                    int(reference_search) + 1,
                ):
                    reference = _extract_square(
                        reference_gray,
                        float(reference_x)
                        + reference_dx,
                        float(reference_y)
                        + reference_dy,
                        int(outer_radius),
                    )

                    if reference is None:
                        continue

                    reference_annulus = reference[
                        annulus_mask
                    ]

                    if reference_annulus.size == 0:
                        continue

                    brightness_shift = float(
                        np.median(
                            reference_annulus
                        )
                        - np.median(
                            current_annulus
                        )
                    )

                    adjusted_current = np.clip(
                        current
                        + brightness_shift,
                        0.0,
                        255.0,
                    )

                    annulus_error = float(
                        np.mean(
                            np.abs(
                                reference[
                                    annulus_mask
                                ]
                                - adjusted_current[
                                    annulus_mask
                                ]
                            )
                        )
                    )

                    if (
                        best_annulus_error is None
                        or annulus_error
                        < best_annulus_error
                    ):
                        best_annulus_error = (
                            annulus_error
                        )
                        best_reference = (
                            reference
                        )
                        best_current = (
                            adjusted_current
                        )

    if (
        best_reference is None
        or best_current is None
    ):
        return 0.0

    reference_blur = cv2.GaussianBlur(
        best_reference,
        (3, 3),
        0,
    )

    current_blur = cv2.GaussianBlur(
        best_current,
        (3, 3),
        0,
    )

    extra_dark = np.clip(
        reference_blur
        - current_blur,
        0.0,
        255.0,
    )

    core_extra = extra_dark[
        core_mask
    ]

    if core_extra.size == 0:
        return 0.0

    mean_extra = float(
        np.mean(
            core_extra
        )
        / 255.0
    )

    strong_fraction = float(
        np.mean(
            core_extra
            >= 28.0
        )
    )

    very_strong_fraction = float(
        np.mean(
            core_extra
            >= 55.0
        )
    )

    return float(
        0.45 * mean_extra
        + 0.35 * strong_fraction
        + 0.20 * very_strong_fraction
    )


def _cluster_1d(values: List[float], k: int) -> List[float] | None:
    if len(values) < k:
        return None

    data = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        60,
        0.10,
    )

    _compactness, _labels, centers = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    return sorted(float(value) for value in centers.reshape(-1))


def _hough_circles(
    gray: np.ndarray,
    bounds: Tuple[int, int, int, int],
    *,
    bubble_radius: int,
    min_dist: int = 13,
) -> List[Tuple[float, float, float]]:
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = bounds

    x0 = max(0, int(round(x0)))
    y0 = max(0, int(round(y0)))
    x1 = min(w - 1, int(round(x1)))
    y1 = min(h - 1, int(round(y1)))

    if x1 <= x0 or y1 <= y0:
        return []

    roi = gray[y0:y1 + 1, x0:x1 + 1]
    roi = cv2.GaussianBlur(roi, (3, 3), 0)

    min_radius = max(4, int(round(bubble_radius * 0.55)))
    max_radius = max(min_radius + 2, int(round(bubble_radius * 1.30)))

    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(10, int(min_dist)),
        param1=100,
        param2=13,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []

    result = []
    for x, y, radius in circles[0]:
        result.append(
            (
                float(x0 + x),
                float(y0 + y),
                float(radius),
            )
        )
    return result


def _validate_cluster_centres(
    actual: List[float] | None,
    expected: List[float],
    max_delta: float,
) -> bool:
    if actual is None or len(actual) != len(expected):
        return False

    return max(
        abs(float(a) - float(b))
        for a, b in zip(actual, sorted(float(v) for v in expected))
    ) <= float(max_delta)


def _calibrate_mcq_grid(
    gray: np.ndarray,
    section: Dict[str, Any],
    template: Dict[str, Any],
) -> Tuple[Dict[str, float], List[float], Dict[str, Any]]:
    options = section.get("options", ["A", "B", "C", "D"])
    expected_x = [float(section["option_x"][option]) for option in options]
    expected_y = [float(v) for v in section["question_y_positions"]]

    bubble_radius = int(template.get("bubble_radius", 10))
    margin = int(template.get("jee_grid_hough_margin", 20))
    max_delta = float(template.get("jee_grid_max_calibration_delta", 22))

    circles = _hough_circles(
        gray,
        (
            min(expected_x) - margin,
            min(expected_y) - margin,
            max(expected_x) + margin,
            max(expected_y) + margin,
        ),
        bubble_radius=bubble_radius,
        min_dist=max(11, int(round(bubble_radius * 1.35))),
    )

    actual_x = None
    actual_y = None

    if len(circles) >= max(24, int(len(expected_x) * len(expected_y) * 0.60)):
        actual_x = _cluster_1d([point[0] for point in circles], len(expected_x))
        actual_y = _cluster_1d([point[1] for point in circles], len(expected_y))

    calibrated = (
        _validate_cluster_centres(actual_x, expected_x, max_delta)
        and _validate_cluster_centres(actual_y, expected_y, max_delta)
    )

    if not calibrated:
        actual_x = sorted(expected_x)
        actual_y = sorted(expected_y)

    x_by_option = {
        option: float(actual_x[index])
        for index, option in enumerate(options)
    }

    return x_by_option, list(actual_y), {
        "calibrated": bool(calibrated),
        "circle_count": len(circles),
        "x_centres": [round(float(v), 2) for v in actual_x],
        "y_centres": [round(float(v), 2) for v in actual_y],
    }


def _classify_mcq(
    gray: np.ndarray,
    coordinates: Dict[str, Tuple[float, float]],
    template: Dict[str, Any],
) -> Dict[str, Any]:
    options = list(coordinates.keys())

    core_radius = int(template.get("jee_core_radius", 6))
    dark_threshold = int(template.get("jee_core_dark_threshold", 140))
    blank_threshold = float(template.get("jee_core_blank_threshold", 0.66))
    filled_threshold = float(template.get("jee_core_filled_threshold", 0.78))
    minimum_gap = float(template.get("jee_core_minimum_gap", 0.10))
    relaxed_threshold = float(template.get("jee_core_relaxed_threshold", 0.72))
    strong_gap = float(template.get("jee_core_strong_gap", 0.20))

    scores = {
        option: _core_fill_ratio(
            gray,
            coordinates[option][0],
            coordinates[option][1],
            radius=core_radius,
            dark_threshold=dark_threshold,
        )
        for option in options
    }

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_option, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence_gap = float(top_score - second_score)

    filled = [
        option
        for option, score in scores.items()
        if score >= filled_threshold
    ]

    if len(filled) >= 2:
        answer = "MULTIPLE"
    elif (
        top_score >= filled_threshold
        and confidence_gap >= minimum_gap
    ):
        answer = top_option
    elif (
        top_score >= relaxed_threshold
        and confidence_gap >= strong_gap
    ):
        answer = top_option
    elif top_score < blank_threshold:
        answer = "BLANK"
    else:
        answer = "UNCERTAIN"

    return {
        "answer": answer,
        "scores": {key: round(float(value), 4) for key, value in scores.items()},
        "highest_score": round(float(top_score), 4),
        "confidence_gap": round(float(confidence_gap), 4),
        "multiple_options": filled if answer == "MULTIPLE" else [],
        "selected_center": (
            [
                int(round(coordinates[top_option][0])),
                int(round(coordinates[top_option][1])),
            ]
            if answer in options or answer == "UNCERTAIN"
            else None
        ),
        "option_centres": {
            option: [
                int(round(coordinates[option][0])),
                int(round(coordinates[option][1])),
            ]
            for option in options
        },
    }


def scan_jee_mcq_sections_robust(
    corrected_image: np.ndarray,
    template: Dict[str, Any],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    gray = _gray(corrected_image)
    detected: Dict[int, Dict[str, Any]] = {}
    calibration_debug: Dict[str, Any] = {}

    for section_index, section in enumerate(template.get("mcq_sections", []), start=1):
        options = section.get("options", ["A", "B", "C", "D"])
        x_by_option, actual_y, debug = _calibrate_mcq_grid(gray, section, template)

        calibration_debug[str(section_index)] = debug

        start_question = int(section["start_question"])
        total_questions = int(section["total_questions"])

        if len(actual_y) != total_questions:
            actual_y = [float(v) for v in section["question_y_positions"]]

        for row_index in range(total_questions):
            question_number = start_question + row_index
            y = float(actual_y[row_index])
            coordinates = {
                option: (float(x_by_option[option]), y)
                for option in options
            }
            record = _classify_mcq(gray, coordinates, template)
            record["grid_calibrated"] = bool(debug["calibrated"])
            detected[question_number] = record

    return detected, calibration_debug


def _calibrate_numerical_question(
    gray: np.ndarray,
    question: Dict[str, Any],
    template: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    columns = question["columns"]
    expected_x = [float(column["x"]) for column in columns]
    expected_y = [float(v) for v in columns[0]["y_positions"]]

    decimal_points = question.get("decimal_points", [])
    expected_decimal_x = [float(item["x"]) for item in decimal_points]
    decimal_y = float(decimal_points[0]["y"]) if decimal_points else min(expected_y) - 40.0

    sign = question.get("sign")
    sign_y = float(sign["y"]) if sign else max(expected_y) + 35.0

    bubble_radius = int(template.get("bubble_radius", 10))
    margin = int(template.get("jee_grid_hough_margin", 20))
    max_delta = float(template.get("jee_grid_max_calibration_delta", 22))

    circles = _hough_circles(
        gray,
        (
            min(expected_x) - margin,
            decimal_y - margin,
            max(expected_x) + margin,
            sign_y + margin,
        ),
        bubble_radius=bubble_radius,
        min_dist=max(11, int(round(bubble_radius * 1.25))),
    )

    digit_candidates = [
        point
        for point in circles
        if min(expected_y) - 16 <= point[1] <= max(expected_y) + 16
    ]

    actual_x = None
    actual_y = None

    if len(digit_candidates) >= max(
        40,
        int(len(expected_x) * len(expected_y) * 0.58),
    ):
        actual_x = _cluster_1d([point[0] for point in digit_candidates], len(expected_x))
        actual_y = _cluster_1d([point[1] for point in digit_candidates], len(expected_y))

    calibrated = (
        _validate_cluster_centres(actual_x, expected_x, max_delta)
        and _validate_cluster_centres(actual_y, expected_y, max_delta)
    )

    if not calibrated:
        actual_x = sorted(expected_x)
        actual_y = sorted(expected_y)

    updated = dict(question)
    updated["columns"] = []

    for index, column in enumerate(columns):
        new_column = dict(column)
        new_column["reference_x"] = float(column["x"])
        new_column["reference_y_positions"] = [
            float(v)
            for v in column["y_positions"]
        ]
        new_column["x"] = float(actual_x[index])
        new_column["y_positions"] = [float(v) for v in actual_y]
        updated["columns"].append(new_column)

    # Calibrate decimal row from detected circles, but only when all five
    # expected decimal circles are clearly present.
    actual_decimal_x = list(expected_decimal_x)
    decimal_candidates = [
        point
        for point in circles
        if abs(point[1] - decimal_y) <= 15
    ]

    if expected_decimal_x and len(decimal_candidates) >= len(expected_decimal_x):
        clustered_decimal_x = _cluster_1d(
            [point[0] for point in decimal_candidates],
            len(expected_decimal_x),
        )
        if _validate_cluster_centres(
            clustered_decimal_x,
            expected_decimal_x,
            max_delta,
        ):
            actual_decimal_x = list(clustered_decimal_x)

    updated["decimal_points"] = [
        {
            **decimal,
            "reference_x": float(decimal["x"]),
            "reference_y": float(decimal["y"]),
            "x": float(actual_decimal_x[index]),
            "y": float(decimal_y),
        }
        for index, decimal in enumerate(decimal_points)
    ]

    if sign:
        sign_candidates = [
            point
            for point in circles
            if abs(point[1] - sign_y) <= 15
            and abs(point[0] - float(sign["x"])) <= max_delta
        ]

        if sign_candidates:
            selected_sign = min(
                sign_candidates,
                key=lambda point:
                abs(point[0] - float(sign["x"]))
                + abs(point[1] - sign_y),
            )
            updated["sign"] = {
                **sign,
                "reference_x": float(sign["x"]),
                "reference_y": float(sign["y"]),
                "x": float(selected_sign[0]),
                "y": float(selected_sign[1]),
            }

    if sign and "reference_x" not in updated.get("sign", {}):
        updated["sign"] = {
            **sign,
            "reference_x": float(sign["x"]),
            "reference_y": float(sign["y"]),
        }

    return updated, {
        "calibrated": bool(calibrated),
        "circle_count": len(circles),
        "digit_circle_count": len(digit_candidates),
        "x_centres": [round(float(v), 2) for v in actual_x],
        "y_centres": [round(float(v), 2) for v in actual_y],
    }


def _classify_numerical_column(
    gray: np.ndarray,
    column: Dict[str, Any],
    template: Dict[str, Any],
) -> Dict[str, Any]:
    values = [
        str(value)
        for value in column.get(
            "values",
            list(range(10)),
        )
    ]

    y_positions = [
        float(value)
        for value in column[
            "y_positions"
        ]
    ]

    reference_y_positions = [
        float(value)
        for value in column.get(
            "reference_y_positions",
            y_positions,
        )
    ]

    x = float(
        column["x"]
    )

    reference_x = float(
        column.get(
            "reference_x",
            x,
        )
    )

    reference_gray = (
        _get_jee_reference_gray(
            template
        )
    )

    core_radius = int(
        template.get(
            "jee_numeric_delta_core_radius",
            5,
        )
    )

    outer_radius = int(
        template.get(
            "jee_numeric_delta_outer_radius",
            9,
        )
    )

    reference_search = int(
        template.get(
            "jee_numeric_reference_search",
            2,
        )
    )

    actual_search = int(
        template.get(
            "jee_numeric_actual_search",
            2,
        )
    )

    scores = {
        value: _extra_ink_score(
            gray,
            reference_gray,
            x,
            y_positions[index],
            reference_x,
            reference_y_positions[index],
            core_radius=core_radius,
            outer_radius=outer_radius,
            reference_search=reference_search,
            actual_search=actual_search,
        )
        for index, value
        in enumerate(values)
    }

    ranked = sorted(
        scores.items(),
        key=lambda item:
        item[1],
        reverse=True,
    )

    top_value, top_score = (
        ranked[0]
    )

    second_score = float(
        ranked[1][1]
        if len(ranked) > 1
        else 0.0
    )

    score_values = np.asarray(
        list(scores.values()),
        dtype=np.float32,
    )

    column_baseline = float(
        np.median(
            score_values
        )
    )

    column_mad = float(
        np.median(
            np.abs(
                score_values
                - column_baseline
            )
        )
    )

    signal = float(
        top_score
        - column_baseline
    )

    gap = float(
        top_score
        - second_score
    )

    blank_signal = float(
        template.get(
            "jee_numeric_adaptive_blank_signal",
            0.028,
        )
    )

    filled_signal = float(
        template.get(
            "jee_numeric_adaptive_filled_signal",
            0.070,
        )
    )

    strong_signal = float(
        template.get(
            "jee_numeric_adaptive_strong_signal",
            0.120,
        )
    )

    minimum_gap = float(
        template.get(
            "jee_numeric_adaptive_minimum_gap",
            0.014,
        )
    )

    strong_minimum_gap = float(
        template.get(
            "jee_numeric_adaptive_strong_minimum_gap",
            0.005,
        )
    )

    dynamic_signal = max(
        filled_signal,
        column_mad
        * float(
            template.get(
                "jee_numeric_adaptive_mad_multiplier",
                3.0,
            )
        ),
    )

    if signal < blank_signal:
        detected_value = ""

    elif (
        signal >= strong_signal
        and gap
        >= strong_minimum_gap
    ):
        detected_value = (
            top_value
        )

    elif (
        signal >= dynamic_signal
        and gap >= minimum_gap
    ):
        detected_value = (
            top_value
        )

    else:
        detected_value = "?"

    selected_index = (
        values.index(
            top_value
        )
    )

    return {
        "value":
            detected_value,

        "top_value":
            top_value,

        "best_score":
            round(
                float(top_score),
                4,
            ),

        "second_score":
            round(
                float(second_score),
                4,
            ),

        "confidence_gap":
            round(
                float(gap),
                4,
            ),

        "baseline_score":
            round(
                float(
                    column_baseline
                ),
                4,
            ),

        "mad":
            round(
                float(
                    column_mad
                ),
                4,
            ),

        "signal":
            round(
                float(signal),
                4,
            ),

        "scores": {
            key: round(
                float(score),
                4,
            )
            for key, score
            in scores.items()
        },

        "center": [
            int(round(x)),
            int(
                round(
                    y_positions[
                        selected_index
                    ]
                )
            ),
        ],

        "score_mode":
            "blank_reference_adaptive_v4",
    }


def detect_numerical_value_robust(
    gray: np.ndarray,
    question: Dict[str, Any],
    template: Dict[str, Any],
) -> Dict[str, Any]:
    columns = question.get("columns", [])
    if not columns:
        return {"answer": "BLANK", "columns": []}

    detected_digits: List[str] = []
    column_details: List[Dict[str, Any]] = []

    for column_index, column in enumerate(columns, start=1):
        detail = _classify_numerical_column(gray, column, template)
        detail["column"] = column_index
        detected_digits.append(detail["value"])
        column_details.append(detail)

    all_scores = [
        float(score)
        for detail in column_details
        for score in detail.get(
            "scores",
            {},
        ).values()
    ]

    question_noise_floor = float(
        np.median(
            np.asarray(
                all_scores,
                dtype=np.float32,
            )
        )
        if all_scores
        else 0.0
    )

    rescue_signal = float(
        template.get(
            "jee_numeric_question_rescue_signal",
            0.050,
        )
    )

    rescue_gap = float(
        template.get(
            "jee_numeric_question_rescue_gap",
            0.008,
        )
    )

    for index, detail in enumerate(
        column_details
    ):
        if detail.get("value") != "?":
            continue

        global_signal = (
            float(
                detail.get(
                    "best_score",
                    0.0,
                )
            )
            - question_noise_floor
        )

        local_signal = float(
            detail.get(
                "signal",
                0.0,
            )
        )

        gap = float(
            detail.get(
                "confidence_gap",
                0.0,
            )
        )

        if (
            max(
                global_signal,
                local_signal,
            )
            >= rescue_signal
            and gap >= rescue_gap
        ):
            rescued = str(
                detail.get(
                    "top_value",
                    "?",
                )
            )

            detail["value"] = (
                rescued
            )

            detail[
                "rescued_from_uncertain"
            ] = True

            detected_digits[
                index
            ] = rescued

    core_radius = int(template.get("jee_core_radius", 6))
    dark_threshold = int(template.get("jee_core_dark_threshold", 140))
    special_threshold = float(
        template.get(
            "jee_numeric_delta_special_threshold",
            0.16,
        )
    )

    decimal_details = []
    decimal_ranked = []

    for decimal in question.get("decimal_points", []):
        score = _extra_ink_score(
            gray,
            _get_jee_reference_gray(template),
            decimal["x"],
            decimal["y"],
            decimal.get("reference_x", decimal["x"]),
            decimal.get("reference_y", decimal["y"]),
            core_radius=int(
                template.get("jee_numeric_delta_core_radius", 5)
            ),
            outer_radius=int(
                template.get("jee_numeric_delta_outer_radius", 9)
            ),
            reference_search=int(
                template.get("jee_numeric_reference_search", 2)
            ),
        )
        detail = {
            "after_column": int(decimal["after_column"]),
            "score": round(float(score), 4),
            "center": [
                int(round(float(decimal["x"]))),
                int(round(float(decimal["y"]))),
            ],
        }
        decimal_details.append(detail)
        decimal_ranked.append(
            (
                detail["after_column"],
                float(score),
            )
        )

    decimal_ranked.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    selected_decimal = None

    if decimal_ranked:
        best_decimal = decimal_ranked[0]
        second_decimal_score = (
            decimal_ranked[1][1]
            if len(decimal_ranked) > 1
            else 0.0
        )

        decimal_gap = (
            float(best_decimal[1])
            - float(second_decimal_score)
        )

        decimal_minimum_gap = float(
            template.get(
                "jee_numeric_decimal_gap",
                0.08,
            )
        )

        decimal_scores = np.asarray(
            [
                float(item[1])
                for item in decimal_ranked
            ],
            dtype=np.float32,
        )

        decimal_baseline = float(
            np.median(
                decimal_scores
            )
            if decimal_scores.size
            else question_noise_floor
        )

        decimal_signal = float(
            best_decimal[1]
            - min(
                decimal_baseline,
                question_noise_floor,
            )
        )

        adaptive_decimal_signal = float(
            template.get(
                "jee_numeric_adaptive_decimal_signal",
                0.045,
            )
        )

        adaptive_decimal_gap = float(
            template.get(
                "jee_numeric_adaptive_decimal_gap",
                0.010,
            )
        )

        if (
            decimal_signal
            >= adaptive_decimal_signal
            and decimal_gap
            >= adaptive_decimal_gap
        ):
            selected_decimal = int(
                best_decimal[0]
            )

    sign_detail = None
    negative = False
    sign = question.get("sign")

    if sign:
        sign_score = _extra_ink_score(
            gray,
            _get_jee_reference_gray(template),
            sign["x"],
            sign["y"],
            sign.get("reference_x", sign["x"]),
            sign.get("reference_y", sign["y"]),
            core_radius=int(
                template.get("jee_numeric_delta_core_radius", 5)
            ),
            outer_radius=int(
                template.get("jee_numeric_delta_outer_radius", 9)
            ),
            reference_search=int(
                template.get("jee_numeric_reference_search", 2)
            ),
        )
        sign_signal = float(
            sign_score
            - question_noise_floor
        )

        negative = (
            sign_signal
            >= float(
                template.get(
                    "jee_numeric_adaptive_sign_signal",
                    0.050,
                )
            )
        )
        sign_detail = {
            "value": "-" if negative else "",
            "score": round(float(sign_score), 4),
            "center": [
                int(round(float(sign["x"]))),
                int(round(float(sign["y"]))),
            ],
        }

    if all(value == "" for value in detected_digits):
        answer = "BLANK"

    elif any(value == "?" for value in detected_digits):
        answer = "UNCERTAIN"

    else:
        used_indices = [
            index
            for index, value in enumerate(detected_digits)
            if value
        ]

        if not used_indices:
            answer = "BLANK"
        else:
            first = min(used_indices)
            last = max(used_indices)
            used_digits = detected_digits[first:last + 1]

            if any(value == "" for value in used_digits):
                answer = "UNCERTAIN"
            else:
                answer = "".join(used_digits)

                if selected_decimal is not None:
                    insertion = selected_decimal - first

                    if insertion <= 0 or insertion >= len(answer):
                        answer = "UNCERTAIN"
                    else:
                        answer = (
                            answer[:insertion]
                            + "."
                            + answer[insertion:]
                        )

                if negative and answer != "UNCERTAIN":
                    answer = "-" + answer

    return {
        "answer": answer,
        "columns": column_details,
        "decimal_points": decimal_details,
        "selected_decimal": selected_decimal,
        "sign": sign_detail,
        "question_noise_floor": round(
            float(
                question_noise_floor
            ),
            4,
        ),
        "reader": "jee_adaptive_reference_reader_v4",
    }


def scan_jee_numerical_sections_robust(
    corrected_image: np.ndarray,
    template: Dict[str, Any],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    gray = _gray(corrected_image)
    detected: Dict[int, Dict[str, Any]] = {}
    debug: Dict[str, Any] = {}
    layout = template.get("numerical_layout", {})

    for section_index, section in enumerate(template.get("numerical_sections", []), start=1):
        questions = section.get("questions")

        if questions is None:
            start_question = int(section["start_question"])
            digit_offsets = [int(value) for value in layout["digit_offsets"]]
            decimal_offsets = [int(value) for value in layout["decimal_offsets"]]
            decimal_after = [int(value) for value in layout["decimal_after_columns"]]
            digit_values = [str(value) for value in layout["digit_values"]]
            digit_y_positions = [int(value) for value in section["digit_y_positions"]]

            questions = []

            for index, base_x_value in enumerate(section["question_x_positions"]):
                base_x = int(base_x_value)

                questions.append(
                    {
                        "question": start_question + index,
                        "columns": [
                            {
                                "x": base_x + offset,
                                "y_positions": digit_y_positions,
                                "values": digit_values,
                            }
                            for offset in digit_offsets
                        ],
                        "decimal_points": [
                            {
                                "x": base_x + offset,
                                "y": int(section["decimal_y"]),
                                "after_column": after_column,
                            }
                            for offset, after_column in zip(
                                decimal_offsets,
                                decimal_after,
                            )
                        ],
                        "sign": {
                            "x": base_x + int(layout.get("sign_offset", 0)),
                            "y": int(section["sign_y"]),
                        },
                    }
                )

        for question in questions:
            question_number = int(question["question"])
            calibrated_question, q_debug = _calibrate_numerical_question(
                gray,
                question,
                template,
            )

            record = detect_numerical_value_robust(
                gray,
                calibrated_question,
                template,
            )

            record["grid_calibrated"] = bool(q_debug["calibrated"])
            detected[question_number] = record
            debug[str(question_number)] = q_debug

    return detected, debug


def scan_jee_answers_robust(
    corrected_image: np.ndarray,
    template: Dict[str, Any],
) -> Dict[str, Any]:
    mcq, mcq_debug = scan_jee_mcq_sections_robust(
        corrected_image,
        template,
    )

    numerical, numerical_debug = scan_jee_numerical_sections_robust(
        corrected_image,
        template,
    )

    return {
        "mcq": mcq,
        "numerical": numerical,
        "_calibration": {
            "mcq": mcq_debug,
            "numerical": numerical_debug,
        },
    }
