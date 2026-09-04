
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
    reference_coordinates: Dict[str, Tuple[float, float]],
    template: Dict[str, Any],
) -> Dict[str, Any]:
    options = list(coordinates.keys())
    reference_gray = _get_jee_reference_gray(template)

    core_radius = int(template.get("jee_mcq_delta_core_radius", 5))
    outer_radius = int(template.get("jee_mcq_delta_outer_radius", 9))
    actual_search = int(template.get("jee_mcq_delta_actual_search", 2))
    reference_search = int(template.get("jee_mcq_delta_reference_search", 2))

    blank_threshold = float(template.get("jee_mcq_delta_blank_threshold", 0.055))
    filled_threshold = float(template.get("jee_mcq_delta_filled_threshold", 0.145))
    multiple_threshold = float(template.get("jee_mcq_delta_multiple_threshold", filled_threshold))
    minimum_gap = float(template.get("jee_mcq_delta_minimum_gap", 0.035))
    relaxed_threshold = float(template.get("jee_mcq_delta_relaxed_threshold", 0.105))
    strong_gap = float(template.get("jee_mcq_delta_strong_gap", 0.055))

    scores = {}
    for option in options:
        actual_x, actual_y = coordinates[option]
        reference_x, reference_y = reference_coordinates[option]
        scores[option] = _extra_ink_score(
            gray,
            reference_gray,
            actual_x,
            actual_y,
            reference_x,
            reference_y,
            core_radius=core_radius,
            outer_radius=outer_radius,
            reference_search=reference_search,
            actual_search=actual_search,
        )

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_option, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence_gap = float(top_score - second_score)

    filled = [
        option
        for option, score in scores.items()
        if score >= multiple_threshold
    ]

    if len(filled) >= 2:
        answer = "MULTIPLE"
    elif top_score >= filled_threshold and confidence_gap >= minimum_gap:
        answer = top_option
    elif top_score >= relaxed_threshold and confidence_gap >= strong_gap:
        answer = top_option
    elif top_score < blank_threshold:
        answer = "BLANK"
    else:
        answer = "UNCERTAIN"

    return {
        "answer": answer,
        "scores": {key: round(float(value), 4) for key, value in scores.items()},
        "highest_score": round(float(top_score), 4),
        "confidence_gap": round(confidence_gap, 4),
        "multiple_options": filled if answer == "MULTIPLE" else [],
        "reader": "jee_mcq_reference_delta_v10_4",
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
            reference_coordinates = {
                option: (
                    float(section["option_x"][option]),
                    float(section["question_y_positions"][row_index]),
                )
                for option in options
            }
            record = _classify_mcq(
                gray,
                coordinates,
                reference_coordinates,
                template,
            )
            record["grid_calibrated"] = bool(debug["calibrated"])
            detected[question_number] = record

    return detected, calibration_debug


def _calibrate_numerical_question(
    gray: np.ndarray,
    question: Dict[str, Any],
    template: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    columns = question["columns"]

    expected_x = [
        float(column["x"])
        for column in columns
    ]

    expected_y = [
        float(value)
        for value in columns[0]["y_positions"]
    ]

    decimal_points = question.get(
        "decimal_points",
        [],
    )

    decimal_y = (
        float(decimal_points[0]["y"])
        if decimal_points
        else min(expected_y) - 40.0
    )

    sign = question.get("sign")

    sign_y = (
        float(sign["y"])
        if sign
        else max(expected_y) + 35.0
    )

    bubble_radius = int(
        template.get(
            "bubble_radius",
            10,
        )
    )

    margin = int(
        template.get(
            "jee_grid_hough_margin",
            20,
        )
    )

    max_delta = float(
        template.get(
            "jee_grid_max_calibration_delta",
            22,
        )
    )

    search_margin = max(
        margin,
        int(round(max_delta + 8.0)),
    )

    circles = _hough_circles(
        gray,
        (
            min(expected_x) - search_margin,
            decimal_y - search_margin,
            max(expected_x) + search_margin,
            sign_y + search_margin,
        ),
        bubble_radius=bubble_radius,
        min_dist=max(
            11,
            int(round(bubble_radius * 1.25)),
        ),
    )

    row_tolerance = max(
        4.5,
        float(bubble_radius) * 0.65,
    )

    broad_y_min = (
        min(expected_y)
        - max_delta
        - 8.0
    )

    broad_y_max = (
        max(expected_y)
        + max_delta
        + 8.0
    )

    broad_candidates = [
        point
        for point in circles
        if broad_y_min <= float(point[1]) <= broad_y_max
    ]

    row_groups: List[List[Tuple[float, float, float]]] = []

    for point in sorted(
        broad_candidates,
        key=lambda item: float(item[1]),
    ):
        if not row_groups:
            row_groups.append([point])
            continue

        current = row_groups[-1]

        current_y = float(
            np.mean(
                [
                    float(item[1])
                    for item in current
                ]
            )
        )

        if abs(float(point[1]) - current_y) <= row_tolerance:
            current.append(point)
        else:
            row_groups.append([point])

    expected_x_span = max(expected_x) - min(expected_x)

    valid_digit_rows = []

    for group in row_groups:
        xs = [
            float(point[0])
            for point in group
        ]

        if len(xs) < max(
            5,
            len(expected_x) - 2,
        ):
            continue

        x_span = max(xs) - min(xs)

        if x_span < expected_x_span * 0.70:
            continue

        valid_digit_rows.append(group)

    if len(valid_digit_rows) > len(expected_y):
        valid_digit_rows = sorted(
            valid_digit_rows,
            key=lambda group: (
                -len(group),
                abs(
                    (
                        max(float(point[0]) for point in group)
                        - min(float(point[0]) for point in group)
                    )
                    - expected_x_span
                ),
            ),
        )[:len(expected_y)]

        valid_digit_rows = sorted(
            valid_digit_rows,
            key=lambda group:
                float(
                    np.mean(
                        [
                            float(point[1])
                            for point in group
                        ]
                    )
                ),
        )

    actual_x = None
    actual_y = None
    digit_candidates: List[Tuple[float, float, float]] = []

    if len(valid_digit_rows) == len(expected_y):
        digit_candidates = [
            point
            for group in valid_digit_rows
            for point in group
        ]

        actual_x = _cluster_1d(
            [
                float(point[0])
                for point in digit_candidates
            ],
            len(expected_x),
        )

        actual_y = [
            float(
                np.median(
                    [
                        float(point[1])
                        for point in group
                    ]
                )
            )
            for group in valid_digit_rows
        ]

    calibrated = (
        _validate_cluster_centres(
            actual_x,
            expected_x,
            max_delta,
        )
        and _validate_cluster_centres(
            actual_y,
            expected_y,
            max_delta,
        )
    )

    if not calibrated:
        actual_x = sorted(expected_x)
        actual_y = sorted(expected_y)

    if calibrated:
        x_fit = np.polyfit(
            np.asarray(
                expected_x,
                dtype=np.float64,
            ),
            np.asarray(
                actual_x,
                dtype=np.float64,
            ),
            1,
        )

        y_fit = np.polyfit(
            np.asarray(
                expected_y,
                dtype=np.float64,
            ),
            np.asarray(
                actual_y,
                dtype=np.float64,
            ),
            1,
        )

        x_scale = float(x_fit[0])
        x_offset = float(x_fit[1])
        y_scale = float(y_fit[0])
        y_offset = float(y_fit[1])

    else:
        x_scale = 1.0
        x_offset = 0.0
        y_scale = 1.0
        y_offset = 0.0

    def project_local_x(
        value: float,
    ) -> float:
        return (
            x_scale * float(value)
            + x_offset
        )

    def project_local_y(
        value: float,
    ) -> float:
        return (
            y_scale * float(value)
            + y_offset
        )

    updated = dict(question)
    updated["columns"] = []

    for index, column in enumerate(columns):
        new_column = dict(column)

        new_column["reference_x"] = float(
            column["x"]
        )

        new_column["reference_y_positions"] = [
            float(value)
            for value in column["y_positions"]
        ]

        new_column["x"] = float(
            actual_x[index]
        )

        new_column["y_positions"] = [
            float(value)
            for value in actual_y
        ]

        updated["columns"].append(
            new_column
        )

    decimal_match_radius = float(
        template.get(
            "jee_numeric_decimal_match_radius",
            14.0,
        )
    )

    calibrated_decimal_points = []

    for decimal in decimal_points:
        expected_dx = float(
            decimal["x"]
        )

        expected_dy = float(
            decimal["y"]
        )

        translated_dx = project_local_x(
            expected_dx
        )

        translated_dy = project_local_y(
            expected_dy
        )

        nearby = [
            point
            for point in circles
            if (
                abs(
                    float(point[0])
                    - translated_dx
                )
                <= decimal_match_radius
                and abs(
                    float(point[1])
                    - translated_dy
                )
                <= decimal_match_radius
            )
        ]

        if nearby:
            selected_decimal_circle = min(
                nearby,
                key=lambda point:
                    (
                        (
                            float(point[0])
                            - translated_dx
                        ) ** 2
                        + (
                            float(point[1])
                            - translated_dy
                        ) ** 2
                    ),
            )

            actual_dx = float(
                selected_decimal_circle[0]
            )

            actual_dy = float(
                selected_decimal_circle[1]
            )

        else:
            actual_dx = translated_dx
            actual_dy = translated_dy

        calibrated_decimal_points.append(
            {
                **decimal,
                "reference_x": expected_dx,
                "reference_y": expected_dy,
                "translated_x": translated_dx,
                "translated_y": translated_dy,
                "x": actual_dx,
                "y": actual_dy,
            }
        )

    updated[
        "decimal_points"
    ] = calibrated_decimal_points

    if sign:
        expected_sign_x = float(
            sign["x"]
        )

        expected_sign_y = float(
            sign["y"]
        )

        translated_sign_x = project_local_x(
            expected_sign_x
        )

        translated_sign_y = project_local_y(
            expected_sign_y
        )

        sign_candidates = [
            point
            for point in circles
            if (
                abs(
                    float(point[1])
                    - translated_sign_y
                )
                <= max(
                    15.0,
                    decimal_match_radius,
                )
                and abs(
                    float(point[0])
                    - translated_sign_x
                )
                <= max_delta
            )
        ]

        if sign_candidates:
            selected_sign = min(
                sign_candidates,
                key=lambda point:
                    (
                        abs(
                            float(point[0])
                            - translated_sign_x
                        )
                        + abs(
                            float(point[1])
                            - translated_sign_y
                        )
                    ),
            )

            actual_sign_x = float(
                selected_sign[0]
            )

            actual_sign_y = float(
                selected_sign[1]
            )

        else:
            actual_sign_x = translated_sign_x
            actual_sign_y = translated_sign_y

        updated["sign"] = {
            **sign,
            "reference_x": expected_sign_x,
            "reference_y": expected_sign_y,
            "translated_x": translated_sign_x,
            "translated_y": translated_sign_y,
            "x": actual_sign_x,
            "y": actual_sign_y,
        }

    return updated, {
        "calibrated": bool(calibrated),
        "circle_count": len(circles),
        "digit_circle_count": len(digit_candidates),
        "digit_row_count": len(valid_digit_rows),
        "x_centres": [
            round(float(value), 2)
            for value in actual_x
        ],
        "y_centres": [
            round(float(value), 2)
            for value in actual_y
        ],
        "local_x_scale": round(float(x_scale), 6),
        "local_x_offset": round(float(x_offset), 3),
        "local_y_scale": round(float(y_scale), 6),
        "local_y_offset": round(float(y_offset), 3),
        "calibration_version": "local_grid_affine_v10_2",
    }


def _solid_core_metrics(
    gray: np.ndarray,
    x: float,
    y: float,
    template: Dict[str, Any],
) -> Dict[str, float]:
    radius = int(
        template.get(
            "jee_numeric_solid_core_radius",
            5,
        )
    )

    x = int(round(float(x)))
    y = int(round(float(y)))

    h, w = gray.shape[:2]

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    roi = gray[y0:y1, x0:x1]

    if roi.size == 0:
        return {
            "mean": 255.0,
            "std": 0.0,
            "p80": 255.0,
            "p90": 255.0,
            "p95": 255.0,
            "spread": 0.0,
        }

    yy, xx = np.ogrid[y0:y1, x0:x1]

    mask = (
        (xx - x) ** 2
        + (yy - y) ** 2
        <= radius ** 2
    )

    values = roi[mask].astype(np.float32)

    if values.size == 0:
        return {
            "mean": 255.0,
            "std": 0.0,
            "p80": 255.0,
            "p90": 255.0,
            "p95": 255.0,
            "spread": 0.0,
        }

    mean = float(np.mean(values))
    p80 = float(np.percentile(values, 80))
    p90 = float(np.percentile(values, 90))
    p95 = float(np.percentile(values, 95))

    return {
        "mean": mean,
        "std": float(np.std(values)),
        "p80": p80,
        "p90": p90,
        "p95": p95,
        "spread": max(0.0, p90 - mean),
    }


def _decimal_annulus_fill_ratio(
    gray: np.ndarray,
    x: float,
    y: float,
    *,
    inner_radius: int = 2,
    outer_radius: int = 5,
    dark_threshold: int = 150,
) -> float:
    x = int(round(float(x)))
    y = int(round(float(y)))

    inner_radius = max(0, int(inner_radius))
    outer_radius = max(inner_radius + 1, int(outer_radius))

    h, w = gray.shape[:2]

    x0 = max(0, x - outer_radius)
    x1 = min(w, x + outer_radius + 1)
    y0 = max(0, y - outer_radius)
    y1 = min(h, y + outer_radius + 1)

    roi = gray[y0:y1, x0:x1]

    if roi.size == 0:
        return 0.0

    yy, xx = np.ogrid[y0:y1, x0:x1]

    distance_sq = (
        (xx - x) ** 2
        + (yy - y) ** 2
    )

    mask = (
        (distance_sq >= inner_radius ** 2)
        & (distance_sq <= outer_radius ** 2)
    )

    count = int(np.count_nonzero(mask))

    if count <= 0:
        return 0.0

    dark = int(
        np.count_nonzero(
            (roi < dark_threshold)
            & mask
        )
    )

    return float(dark) / float(count)



def _decimal_local_contrast_metrics(
    gray: np.ndarray,
    x: float,
    y: float,
    template: Dict[str, Any],
) -> Dict[str, float]:
    # Compare the decimal interior with its own nearby paper background.
    # This makes the decision independent of overall mobile brightness.

    inner_radius = int(
        template.get(
            "jee_numeric_decimal_contrast_inner_radius",
            4,
        )
    )

    background_inner = int(
        template.get(
            "jee_numeric_decimal_background_inner_radius",
            12,
        )
    )

    background_outer = int(
        template.get(
            "jee_numeric_decimal_background_outer_radius",
            15,
        )
    )

    x = int(round(float(x)))
    y = int(round(float(y)))

    h, w = gray.shape[:2]

    radius = max(
        background_outer,
        inner_radius,
    )

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    roi = gray[y0:y1, x0:x1]

    if roi.size == 0:
        return {
            "inner_mean": 255.0,
            "inner_p90": 255.0,
            "background_median": 255.0,
            "contrast_ratio": 1.0,
            "contrast_delta": 0.0,
        }

    yy, xx = np.ogrid[y0:y1, x0:x1]

    distance_sq = (
        (xx - x) ** 2
        + (yy - y) ** 2
    )

    inner_mask = (
        distance_sq
        <= inner_radius ** 2
    )

    background_mask = (
        (distance_sq >= background_inner ** 2)
        & (distance_sq <= background_outer ** 2)
    )

    inner_values = roi[
        inner_mask
    ].astype(np.float32)

    background_values = roi[
        background_mask
    ].astype(np.float32)

    if (
        inner_values.size == 0
        or background_values.size == 0
    ):
        return {
            "inner_mean": 255.0,
            "inner_p90": 255.0,
            "background_median": 255.0,
            "contrast_ratio": 1.0,
            "contrast_delta": 0.0,
        }

    inner_mean = float(
        np.mean(
            inner_values
        )
    )

    inner_p90 = float(
        np.percentile(
            inner_values,
            90,
        )
    )

    background_median = float(
        np.median(
            background_values
        )
    )

    contrast_ratio = (
        inner_mean
        / max(
            background_median,
            1.0,
        )
    )

    contrast_delta = (
        background_median
        - inner_mean
    )

    return {
        "inner_mean":
            inner_mean,

        "inner_p90":
            inner_p90,

        "background_median":
            background_median,

        "contrast_ratio":
            float(contrast_ratio),

        "contrast_delta":
            float(contrast_delta),
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
        for value in column["y_positions"]
    ]

    x = float(column["x"])

    metrics = []

    for index, value in enumerate(values):
        measurement = _solid_core_metrics(
            gray,
            x,
            y_positions[index],
            template,
        )

        metrics.append(
            {
                "value": value,
                "center": [
                    int(round(x)),
                    int(round(y_positions[index])),
                ],
                **measurement,
            }
        )

    p90_values = np.asarray(
        [item["p90"] for item in metrics],
        dtype=np.float32,
    )

    std_values = np.asarray(
        [item["std"] for item in metrics],
        dtype=np.float32,
    )

    median_p90 = float(np.median(p90_values))
    median_std = float(np.median(std_values))

    p90_ratio_limit = float(
        template.get(
            "jee_numeric_uniform_p90_ratio",
            0.62,
        )
    )

    std_ratio_limit = float(
        template.get(
            "jee_numeric_uniform_std_ratio",
            0.50,
        )
    )

    std_max = float(
        template.get(
            "jee_numeric_uniform_std_max",
            22.0,
        )
    )

    candidates = []

    for item in metrics:
        p90_ratio = (
            float(item["p90"])
            / max(median_p90, 1.0)
        )

        std_ratio = (
            float(item["std"])
            / max(median_std, 1.0)
        )

        item["p90_ratio"] = p90_ratio
        item["std_ratio"] = std_ratio

        if (
            p90_ratio <= p90_ratio_limit
            and std_ratio <= std_ratio_limit
            and float(item["std"]) <= std_max
        ):
            candidates.append(item)

    if len(candidates) == 1:
        selected = candidates[0]
        value = str(selected["value"])
        status = "FILLED"
        center = list(selected["center"])
    elif len(candidates) >= 2:
        value = "?"
        status = "MULTIPLE"
        center = None
    else:
        value = ""
        status = "BLANK"
        center = None

    return {
        "value": value,
        "status": status,
        "center": center,
        "median_p90": round(median_p90, 4),
        "median_std": round(median_std, 4),
        "filled_candidates": [
            {
                "value": str(item["value"]),
                "center": list(item["center"]),
                "mean": round(float(item["mean"]), 4),
                "std": round(float(item["std"]), 4),
                "p90": round(float(item["p90"]), 4),
                "p90_ratio": round(float(item["p90_ratio"]), 4),
                "std_ratio": round(float(item["std_ratio"]), 4),
            }
            for item in candidates
        ],
        "metrics": {
            str(item["value"]): {
                "mean": round(float(item["mean"]), 4),
                "std": round(float(item["std"]), 4),
                "p80": round(float(item["p80"]), 4),
                "p90": round(float(item["p90"]), 4),
                "p95": round(float(item["p95"]), 4),
                "spread": round(float(item["spread"]), 4),
                "p90_ratio": round(float(item["p90_ratio"]), 4),
                "std_ratio": round(float(item["std_ratio"]), 4),
            }
            for item in metrics
        },
        "reader": "jee_solid_core_column_v5",
        "classifier_version": "uniform_core_v6",
    }


def detect_numerical_value_robust(
    gray: np.ndarray,
    question: Dict[str, Any],
    template: Dict[str, Any],
) -> Dict[str, Any]:
    columns = question.get("columns", [])

    if not columns:
        return {
            "answer": "BLANK",
            "columns": [],
            "decimal_points": [],
            "selected_decimal": None,
            "sign": None,
            "reader": "jee_solid_core_reader_v5",
        }

    detected_digits: List[str] = []
    column_details: List[Dict[str, Any]] = []

    blank_cell_means = []
    filled_cell_means = []

    for column_index, column in enumerate(
        columns,
        start=1,
    ):
        detail = _classify_numerical_column(
            gray,
            column,
            template,
        )

        detail["column"] = column_index

        detected_digits.append(
            str(detail["value"])
        )

        column_details.append(detail)

        candidate_values = {
            str(item["value"])
            for item in detail.get(
                "filled_candidates",
                [],
            )
        }

        for digit, metric in detail.get(
            "metrics",
            {},
        ).items():
            mean = float(
                metric.get(
                    "mean",
                    255.0,
                )
            )

            if str(digit) in candidate_values:
                filled_cell_means.append(mean)
            else:
                blank_cell_means.append(mean)

    if blank_cell_means:
        question_blank_mean = float(
            np.median(
                np.asarray(
                    blank_cell_means,
                    dtype=np.float32,
                )
            )
        )
    else:
        all_means = (
            blank_cell_means
            + filled_cell_means
        )

        question_blank_mean = float(
            np.median(
                np.asarray(
                    all_means,
                    dtype=np.float32,
                )
            )
            if all_means
            else 255.0
        )

    special_mean_ratio = float(
        template.get(
            "jee_numeric_special_mean_ratio",
            0.55,
        )
    )

    decimal_details = []
    filled_decimals = []

    contrast_ratio_limit = float(
        template.get(
            "jee_numeric_decimal_contrast_ratio",
            0.68,
        )
    )

    contrast_delta_min = float(
        template.get(
            "jee_numeric_decimal_contrast_delta",
            35.0,
        )
    )

    for decimal in question.get(
        "decimal_points",
        [],
    ):
        base_x = float(
            decimal["x"]
        )

        base_y = float(
            decimal["y"]
        )

        measurement = (
            _decimal_local_contrast_metrics(
                gray,
                base_x,
                base_y,
                template,
            )
        )

        contrast_ratio = float(
            measurement[
                "contrast_ratio"
            ]
        )

        contrast_delta = float(
            measurement[
                "contrast_delta"
            ]
        )

        filled = (
            contrast_ratio
            <= contrast_ratio_limit
            and contrast_delta
            >= contrast_delta_min
        )

        detail = {
            "after_column":
                int(
                    decimal[
                        "after_column"
                    ]
                ),

            "center": [
                int(round(base_x)),
                int(round(base_y)),
            ],

            "filled":
                bool(filled),

            "inner_mean":
                round(
                    float(
                        measurement[
                            "inner_mean"
                        ]
                    ),
                    4,
                ),

            "inner_p90":
                round(
                    float(
                        measurement[
                            "inner_p90"
                        ]
                    ),
                    4,
                ),

            "background_median":
                round(
                    float(
                        measurement[
                            "background_median"
                        ]
                    ),
                    4,
                ),

            "contrast_ratio":
                round(
                    contrast_ratio,
                    4,
                ),

            "contrast_delta":
                round(
                    contrast_delta,
                    4,
                ),

            "score_mode":
                "decimal_local_contrast_v6_4",

            "sampling_mode":
                "local_background_relative_v6_4",

            "legacy_score_mode":
                "decimal_annulus_fill_v6_3",

            "legacy_sampling_mode":
                "individual_hough_center_v6_3",
        }

        decimal_details.append(
            detail
        )

        if filled:
            filled_decimals.append(
                detail
            )

    selected_decimal = (
        int(
            filled_decimals[0][
                "after_column"
            ]
        )
        if len(filled_decimals) == 1
        else None
    )

    if len(filled_decimals) >= 2:
        decimal_status = "MULTIPLE"
    elif len(filled_decimals) == 1:
        decimal_status = "FILLED"
    else:
        decimal_status = "BLANK"

    sign_detail = None
    negative = False

    sign = question.get("sign")

    if sign:
        measurement = _solid_core_metrics(
            gray,
            float(sign["x"]),
            float(sign["y"]),
            template,
        )

        sign_mean_ratio = (
            float(measurement["mean"])
            / max(question_blank_mean, 1.0)
        )

        negative = (
            sign_mean_ratio
            <= special_mean_ratio
        )

        sign_detail = {
            "value":
                "-" if negative else "",

            "filled":
                bool(negative),

            "center": [
                int(round(float(sign["x"]))),
                int(round(float(sign["y"]))),
            ],

            "mean":
                round(
                    float(measurement["mean"]),
                    4,
                ),

            "spread":
                round(
                    float(measurement["spread"]),
                    4,
                ),

            "mean_ratio":
                round(
                    sign_mean_ratio,
                    4,
                ),
        }

    has_multiple_column = any(
        detail.get("status")
        == "MULTIPLE"
        for detail in column_details
    )

    if has_multiple_column:
        answer = "UNCERTAIN"

    elif len(filled_decimals) >= 2:
        answer = "UNCERTAIN"

    elif all(
        value == ""
        for value in detected_digits
    ):
        answer = "BLANK"

    elif any(
        value == "?"
        for value in detected_digits
    ):
        answer = "UNCERTAIN"

    else:
        used_indices = [
            index
            for index, value
            in enumerate(detected_digits)
            if value
        ]

        if not used_indices:
            answer = "BLANK"

        else:
            first = min(used_indices)
            last = max(used_indices)

            used_digits = (
                detected_digits[
                    first:last + 1
                ]
            )

            if any(
                value == ""
                for value in used_digits
            ):
                answer = "UNCERTAIN"

            else:
                answer = "".join(
                    used_digits
                )

                if selected_decimal is not None:
                    insertion = (
                        selected_decimal
                        - first
                    )

                    if (
                        insertion <= 0
                        or insertion
                        >= len(answer)
                    ):
                        answer = "UNCERTAIN"

                    else:
                        answer = (
                            answer[:insertion]
                            + "."
                            + answer[insertion:]
                        )

                if (
                    negative
                    and answer
                    != "UNCERTAIN"
                ):
                    answer = "-" + answer

    return {
        "answer": answer,
        "columns": column_details,
        "decimal_points": decimal_details,
        "decimal_status": decimal_status,
        "selected_decimal": selected_decimal,
        "sign": sign_detail,
        "question_blank_mean":
            round(question_blank_mean, 4),
        "reader":
            "jee_solid_core_reader_v5",

        "numeric_classifier_version":
            "uniform_core_v6",

        "decimal_classifier_version":
            "decimal_core_v6_1",
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
