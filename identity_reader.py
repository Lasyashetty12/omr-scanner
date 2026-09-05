
from __future__ import annotations

from typing import Any, Dict, List, Tuple

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


def _hough(
    gray: np.ndarray,
    bounds: Tuple[int, int, int, int],
    *,
    bubble_radius: int,
) -> List[Tuple[float, float, float]]:
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = bounds

    x0 = max(0, int(round(x0)))
    y0 = max(0, int(round(y0)))
    x1 = min(w - 1, int(round(x1)))
    y1 = min(h - 1, int(round(y1)))

    if x1 <= x0 or y1 <= y0:
        return []

    roi = cv2.GaussianBlur(gray[y0:y1 + 1, x0:x1 + 1], (3, 3), 0)

    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(12, int(round(bubble_radius * 1.35))),
        param1=100,
        param2=13,
        minRadius=max(4, int(round(bubble_radius * 0.55))),
        maxRadius=max(8, int(round(bubble_radius * 1.35))),
    )

    if circles is None:
        return []

    return [
        (float(x0 + x), float(y0 + y), float(radius))
        for x, y, radius in circles[0]
    ]


def _detect_roll_number(
    gray: np.ndarray,
    config: Dict[str, Any],
    template: Dict[str, Any],
) -> Dict[str, Any]:
    expected_x = [float(v) for v in config["x_positions"]]
    expected_y = [float(v) for v in config["y_positions"]]
    values = [str(v) for v in config.get("values", list(range(10)))]

    bubble_radius = int(template.get("bubble_radius", 10))
    margin = int(config.get("hough_margin", 20))
    max_delta = float(config.get("max_calibration_delta", 22))

    circles = _hough(
        gray,
        (
            min(expected_x) - margin,
            min(expected_y) - margin,
            max(expected_x) + margin,
            max(expected_y) + margin,
        ),
        bubble_radius=bubble_radius,
    )

    actual_x = None
    actual_y = None

    minimum_circles = max(
        35,
        int(len(expected_x) * len(expected_y) * 0.50),
    )

    if len(circles) >= minimum_circles:
        actual_x = _cluster_1d(
            [point[0] for point in circles],
            len(expected_x),
        )
        actual_y = _cluster_1d(
            [point[1] for point in circles],
            len(expected_y),
        )

    calibrated = (
        actual_x is not None
        and actual_y is not None
        and max(abs(a - b) for a, b in zip(actual_x, sorted(expected_x))) <= max_delta
        and max(abs(a - b) for a, b in zip(actual_y, sorted(expected_y))) <= max_delta
    )

    if not calibrated:
        actual_x = sorted(expected_x)
        actual_y = sorted(expected_y)

    core_radius = int(config.get("solid_core_radius", 5))
    p90_threshold = float(config.get("solid_p90_threshold", 125.0))
    std_threshold = float(config.get("solid_std_threshold", 34.0))
    relative_p90_ratio = float(config.get("solid_p90_ratio", 0.72))
    minimum_p90_gap = float(config.get("solid_minimum_p90_gap", 16.0))

    def core_stats(x: float, y: float) -> Dict[str, float]:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        radius = max(2, core_radius)
        h, w = gray.shape[:2]

        x0 = max(0, xi - radius)
        x1 = min(w, xi + radius + 1)
        y0 = max(0, yi - radius)
        y1 = min(h, yi + radius + 1)

        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            return {"mean": 255.0, "p90": 255.0, "std": 0.0}

        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = ((xx - xi) ** 2 + (yy - yi) ** 2) <= radius ** 2
        vals = roi[mask].astype(np.float32)

        if vals.size == 0:
            return {"mean": 255.0, "p90": 255.0, "std": 0.0}

        return {
            "mean": float(np.mean(vals)),
            "p90": float(np.percentile(vals, 90)),
            "std": float(np.std(vals)),
        }

    digits = []
    column_details = []

    for column_index, x in enumerate(actual_x):
        metrics = {
            value: core_stats(x, actual_y[index])
            for index, value in enumerate(values)
        }

        p90_values = [float(item["p90"]) for item in metrics.values()]
        median_p90 = float(np.median(p90_values))

        ranked = sorted(
            metrics.items(),
            key=lambda item: (
                float(item[1]["p90"]),
                float(item[1]["std"]),
                float(item[1]["mean"]),
            ),
        )

        best_value, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else {"p90": 255.0}

        best_p90 = float(best["p90"])
        second_p90 = float(second["p90"])
        p90_gap = second_p90 - best_p90

        relative_ok = best_p90 <= median_p90 * relative_p90_ratio
        absolute_ok = (
            best_p90 <= p90_threshold
            and float(best["std"]) <= std_threshold
        )

        digit = (
            best_value
            if absolute_ok and relative_ok and p90_gap >= minimum_p90_gap
            else None
        )

        if digit is None:
            old_scores = {
                value: _core_fill_ratio(
                    gray,
                    x,
                    actual_y[index],
                    radius=int(config.get("core_radius", 6)),
                    dark_threshold=int(config.get("dark_threshold", 140)),
                )
                for index, value in enumerate(values)
            }

            old_ranked = sorted(old_scores.items(), key=lambda item: item[1], reverse=True)
            old_best_value, old_best = old_ranked[0]
            old_second = old_ranked[1][1] if len(old_ranked) > 1 else 0.0

            if (
                old_best >= float(config.get("filled_threshold", 0.76))
                and (old_best - old_second) >= float(config.get("minimum_confidence_gap", 0.12))
                and best_p90 <= p90_threshold + 18.0
            ):
                digit = old_best_value

        digits.append(digit)

        column_details.append(
            {
                "column": column_index + 1,
                "value": digit,
                "best_p90": round(best_p90, 2),
                "second_p90": round(second_p90, 2),
                "p90_gap": round(p90_gap, 2),
                "median_p90": round(median_p90, 2),
                "best_std": round(float(best["std"]), 2),
                "metrics": {
                    value: {
                        key: round(float(metric), 2)
                        for key, metric in details.items()
                    }
                    for value, details in metrics.items()
                },
            }
        )

    complete = all(value is not None for value in digits)

    return {
        "value": "".join(digits) if complete else None,
        "complete": bool(complete),
        "columns": column_details,
        "grid_calibrated": bool(calibrated),
        "circle_count": len(circles),
        "reader": "solid_roll_grid_v10_4",
    }



def _roll_disk_dark_ratio(
    gray: np.ndarray,
    x: float,
    y: float,
    radius: int = 8,
) -> float:
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    radius = max(4, int(radius))

    h, w = gray.shape[:2]
    x0 = max(0, xi - radius)
    x1 = min(w, xi + radius + 1)
    y0 = max(0, yi - radius)
    y1 = min(h, yi + radius + 1)

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (
        (xx - xi) ** 2
        + (yy - yi) ** 2
        <= radius ** 2
    )

    pixels = roi[mask].astype(np.float32)
    if pixels.size == 0:
        return 0.0

    paper = float(np.percentile(roi, 80))
    dark_threshold = float(
        np.clip(
            paper - 38.0,
            85.0,
            165.0,
        )
    )

    return float(
        np.mean(
            pixels < dark_threshold
        )
    )


def _crop_roll_bubble(
    gray: np.ndarray,
    x: float,
    y: float,
    radius: int = 10,
) -> np.ndarray:
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    radius = max(8, int(radius))

    h, w = gray.shape[:2]
    x0 = max(0, xi - radius)
    x1 = min(w, xi + radius + 1)
    y0 = max(0, yi - radius)
    y1 = min(h, yi + radius + 1)

    return gray[y0:y1, x0:x1]


def _detect_roll_number_ml_fallback(
    gray: np.ndarray,
    config: Dict[str, Any],
    template: Dict[str, Any],
) -> Dict[str, Any]:
    from ml_omr.inference import classify_batch

    expected_x = [float(value) for value in config["x_positions"]]
    expected_y = [float(value) for value in config["y_positions"]]
    values = [
        str(value)
        for value in config.get("values", list(range(10)))
    ]

    bubble_radius = int(template.get("bubble_radius", 10))
    margin = int(config.get("hough_margin", 20))
    max_delta = float(config.get("max_calibration_delta", 22))

    circles = _hough(
        gray,
        (
            min(expected_x) - margin,
            min(expected_y) - margin,
            max(expected_x) + margin,
            max(expected_y) + margin,
        ),
        bubble_radius=bubble_radius,
    )

    actual_x = None
    actual_y = None

    minimum_circles = max(
        28,
        int(len(expected_x) * len(expected_y) * 0.40),
    )

    if len(circles) >= minimum_circles:
        actual_x = _cluster_1d(
            [point[0] for point in circles],
            len(expected_x),
        )
        actual_y = _cluster_1d(
            [point[1] for point in circles],
            len(expected_y),
        )

    calibrated = (
        actual_x is not None
        and actual_y is not None
        and max(
            abs(a - b)
            for a, b in zip(actual_x, sorted(expected_x))
        ) <= max_delta
        and max(
            abs(a - b)
            for a, b in zip(actual_y, sorted(expected_y))
        ) <= max_delta
    )

    if not calibrated:
        actual_x = sorted(expected_x)
        actual_y = sorted(expected_y)

    crop_radius = int(config.get("ml_crop_radius", 10))

    crops = []
    locations = []

    for column_index, x in enumerate(actual_x):
        for row_index, value in enumerate(values):
            y = actual_y[row_index]
            crops.append(
                _crop_roll_bubble(
                    gray,
                    x,
                    y,
                    radius=crop_radius,
                )
            )
            locations.append(
                (
                    column_index,
                    row_index,
                    value,
                    x,
                    y,
                )
            )

    predictions = classify_batch(crops)

    by_column: Dict[int, List[Dict[str, Any]]] = {
        index: []
        for index in range(len(actual_x))
    }

    for location, prediction in zip(locations, predictions):
        (
            column_index,
            row_index,
            value,
            x,
            y,
        ) = location

        probabilities = (
            prediction.get("probabilities", {})
            if isinstance(prediction, dict)
            else {}
        )

        filled_probability = float(
            probabilities.get("filled", 0.0)
        )
        blank_probability = float(
            probabilities.get("blank", 0.0)
        )

        disk_ratio = _roll_disk_dark_ratio(
            gray,
            x,
            y,
            radius=int(config.get("ml_disk_radius", 8)),
        )

        combined = (
            0.72 * filled_probability
            + 0.28 * disk_ratio
        )

        by_column[column_index].append(
            {
                "row": row_index,
                "value": value,
                "filled_probability": filled_probability,
                "blank_probability": blank_probability,
                "disk_dark_ratio": disk_ratio,
                "combined_score": combined,
                "center": [
                    int(round(x)),
                    int(round(y)),
                ],
            }
        )

    digits: List[str | None] = []
    column_details = []

    for column_index in range(len(actual_x)):
        ranked = sorted(
            by_column[column_index],
            key=lambda item: (
                float(item["combined_score"]),
                float(item["filled_probability"]),
                float(item["disk_dark_ratio"]),
            ),
            reverse=True,
        )

        best = ranked[0]
        second = ranked[1]

        combined_gap = (
            float(best["combined_score"])
            - float(second["combined_score"])
        )
        ml_gap = (
            float(best["filled_probability"])
            - float(second["filled_probability"])
        )
        disk_gap = (
            float(best["disk_dark_ratio"])
            - float(second["disk_dark_ratio"])
        )

        strong_ml = (
            float(best["filled_probability"]) >= 0.52
            and ml_gap >= 0.10
        )
        strong_disk = (
            float(best["disk_dark_ratio"]) >= 0.46
            and disk_gap >= 0.10
        )
        balanced = (
            float(best["combined_score"]) >= 0.46
            and combined_gap >= 0.085
            and (
                float(best["filled_probability"]) >= 0.42
                or float(best["disk_dark_ratio"]) >= 0.42
            )
        )

        digit = (
            str(best["value"])
            if (
                strong_ml
                or strong_disk
                or balanced
            )
            else None
        )

        digits.append(digit)

        column_details.append(
            {
                "column": column_index + 1,
                "value": digit,
                "best_value": str(best["value"]),
                "best_combined": round(
                    float(best["combined_score"]),
                    4,
                ),
                "combined_gap": round(combined_gap, 4),
                "ml_filled": round(
                    float(best["filled_probability"]),
                    4,
                ),
                "ml_gap": round(ml_gap, 4),
                "disk_dark_ratio": round(
                    float(best["disk_dark_ratio"]),
                    4,
                ),
                "disk_gap": round(disk_gap, 4),
                "center": list(best["center"]),
            }
        )

    complete = all(value is not None for value in digits)

    return {
        "value": (
            "".join(digits)
            if complete
            else None
        ),
        "complete": bool(complete),
        "columns": column_details,
        "grid_calibrated": bool(calibrated),
        "circle_count": len(circles),
        "reader": "jee_roll_ml_disk_v10_14",
    }



def _detect_choice_row(
    gray: np.ndarray,
    config: Dict[str, Any],
    template: Dict[str, Any],
) -> Dict[str, Any]:
    choices = config.get("choices", {})
    if not choices:
        return {"value": None, "scores": {}}

    labels = list(choices.keys())
    expected = [choices[label] for label in labels]

    expected_x = [float(point[0]) for point in expected]
    expected_y = [float(point[1]) for point in expected]
    target_y = float(np.median(expected_y))

    bubble_radius = int(template.get("bubble_radius", 10))
    margin = int(config.get("hough_margin", 20))
    max_delta = float(config.get("max_calibration_delta", 22))

    circles = _hough(
        gray,
        (
            min(expected_x) - margin,
            target_y - margin,
            max(expected_x) + margin,
            target_y + margin,
        ),
        bubble_radius=bubble_radius,
    )

    actual_x = None

    if len(circles) >= len(labels):
        actual_x = _cluster_1d(
            [
                point[0]
                for point in circles
                if abs(point[1] - target_y) <= margin
            ],
            len(labels),
        )

    calibrated = (
        actual_x is not None
        and max(
            abs(a - b)
            for a, b in zip(actual_x, sorted(expected_x))
        ) <= max_delta
    )

    if not calibrated:
        actual_x = sorted(expected_x)

    # Preserve left-to-right label ordering from the template.
    sorted_pairs = sorted(
        zip(labels, expected_x),
        key=lambda item: item[1],
    )

    labels_by_x = [item[0] for item in sorted_pairs]

    core_radius = int(config.get("core_radius", 6))
    dark_threshold = int(config.get("dark_threshold", 140))
    filled_threshold = float(config.get("filled_threshold", 0.68))
    minimum_gap = float(config.get("minimum_confidence_gap", 0.12))

    scores = {
        label: _core_fill_ratio(
            gray,
            actual_x[index],
            target_y,
            radius=core_radius,
            dark_threshold=dark_threshold,
        )
        for index, label in enumerate(labels_by_x)
    }

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = float(best_score - second_score)

    value = (
        best_label
        if best_score >= filled_threshold and gap >= minimum_gap
        else None
    )

    return {
        "value": value,
        "best_score": round(float(best_score), 4),
        "confidence_gap": round(float(gap), 4),
        "scores": {
            key: round(float(score), 4)
            for key, score in scores.items()
        },
        "grid_calibrated": bool(calibrated),
        "circle_count": len(circles),
    }


def detect_identity_fields(
    corrected_image: np.ndarray,
    template: Dict[str, Any],
) -> Dict[str, Any]:
    config = template.get("identity") or {}
    if not config:
        return {}

    gray = _gray(corrected_image)

    result: Dict[str, Any] = {}

    if config.get("roll_number"):
        roll = _detect_roll_number(
            gray,
            config["roll_number"],
            template,
        )

        if not roll.get("value"):
            try:
                ml_roll = (
                    _detect_roll_number_ml_fallback(
                        gray,
                        config["roll_number"],
                        template,
                    )
                )
            except Exception as ml_roll_error:
                ml_roll = {
                    "value": None,
                    "complete": False,
                    "reader":
                        "jee_roll_ml_disk_v10_14",
                    "warning":
                        str(ml_roll_error),
                }

            if ml_roll.get("value"):
                roll = ml_roll
            else:
                roll["ml_fallback"] = ml_roll

        result["roll_number"] = roll.get("value")
        result["roll_number_details"] = roll

    if config.get("class"):
        class_result = _detect_choice_row(
            gray,
            config["class"],
            template,
        )
        result["class"] = class_result.get("value")
        result["class_details"] = class_result

    if config.get("exam"):
        exam_result = _detect_choice_row(
            gray,
            config["exam"],
            template,
        )
        result["exam"] = exam_result.get("value")
        result["exam_details"] = exam_result

    return result
