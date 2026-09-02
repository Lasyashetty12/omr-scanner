
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from jee_reader import _calibrate_numerical_question


Point = Tuple[float, float]
Circle = Tuple[float, float, float]


def _gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("JEE precise reader received an empty image.")
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _hough_circles(
    gray: np.ndarray,
    bounds: Tuple[float, float, float, float],
    bubble_radius: int,
) -> List[Circle]:
    height, width = gray.shape[:2]
    x0, y0, x1, y1 = bounds

    x0 = max(0, int(np.floor(x0)))
    y0 = max(0, int(np.floor(y0)))
    x1 = min(width - 1, int(np.ceil(x1)))
    y1 = min(height - 1, int(np.ceil(y1)))

    if x1 <= x0 or y1 <= y0:
        return []

    roi = gray[y0:y1 + 1, x0:x1 + 1]
    roi = cv2.GaussianBlur(roi, (3, 3), 0)

    min_radius = max(4, int(round(bubble_radius * 0.52)))
    max_radius = max(min_radius + 2, int(round(bubble_radius * 1.45)))

    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(10, int(round(bubble_radius * 1.15))),
        param1=100,
        param2=10,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []

    return [
        (float(x0 + x), float(y0 + y), float(radius))
        for x, y, radius in circles[0]
    ]


def _nearest_offsets(
    circles: Sequence[Circle],
    expected: Sequence[Point],
    max_delta: float,
) -> Tuple[float, float, int]:
    offsets: List[Tuple[float, float]] = []

    for ex, ey in expected:
        best = None

        for cx, cy, _radius in circles:
            dx = cx - ex
            dy = cy - ey
            distance = float(np.hypot(dx, dy))

            if distance > max_delta:
                continue

            if best is None or distance < best[0]:
                best = (distance, dx, dy)

        if best is not None:
            offsets.append((best[1], best[2]))

    if len(offsets) < 2:
        return 0.0, 0.0, len(offsets)

    dx = float(np.median([item[0] for item in offsets]))
    dy = float(np.median([item[1] for item in offsets]))

    return dx, dy, len(offsets)


def _assign_unique(
    circles: Sequence[Circle],
    expected: Sequence[Point],
    max_delta: float,
) -> List[Point]:
    candidates: List[Tuple[float, int, int]] = []

    for expected_index, (ex, ey) in enumerate(expected):
        for circle_index, (cx, cy, _radius) in enumerate(circles):
            distance = float(np.hypot(cx - ex, cy - ey))

            if distance <= max_delta:
                candidates.append(
                    (distance, expected_index, circle_index)
                )

    assigned: Dict[int, Point] = {}
    used_circles = set()

    for _distance, expected_index, circle_index in sorted(candidates):
        if expected_index in assigned:
            continue

        if circle_index in used_circles:
            continue

        circle = circles[circle_index]

        assigned[expected_index] = (
            float(circle[0]),
            float(circle[1]),
        )

        used_circles.add(circle_index)

    return [
        assigned.get(
            index,
            (float(point[0]), float(point[1])),
        )
        for index, point in enumerate(expected)
    ]


def _fill_ratio(
    gray: np.ndarray,
    x: float,
    y: float,
    radius: int,
    dark_threshold: int,
) -> float:
    cx = int(round(float(x)))
    cy = int(round(float(y)))
    radius = max(2, int(radius))

    x0 = max(0, cx - radius)
    x1 = min(gray.shape[1], cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(gray.shape[0], cy + radius + 1)

    roi = gray[y0:y1, x0:x1]

    if roi.size == 0:
        return 0.0

    yy, xx = np.ogrid[y0:y1, x0:x1]

    mask = (
        (xx - cx) ** 2
        + (yy - cy) ** 2
        <= radius ** 2
    )

    total = int(np.count_nonzero(mask))

    if total <= 0:
        return 0.0

    dark = int(
        np.count_nonzero(
            (roi < dark_threshold)
            & mask
        )
    )

    return float(dark) / float(total)


def _numeric_local_fill(
    gray,
    x,
    y,
    radius,
    dark_threshold,
    search_radius=3,
):
    best_score = -1.0
    best_center = (float(x), float(y))
    search_radius = int(search_radius)
    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            if dx * dx + dy * dy > search_radius * search_radius:
                continue
            cx = float(x) + dx
            cy = float(y) + dy
            score = _fill_ratio(
                gray, cx, cy, radius, dark_threshold
            )
            if score > best_score:
                best_score = float(score)
                best_center = (cx, cy)
    return float(best_score), best_center


def scan_jee_mcq_precise(
    corrected_image: np.ndarray,
    template: Dict[str, Any],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    gray = _gray(corrected_image)

    bubble_radius = int(
        template.get("bubble_radius", 10)
    )

    core_radius = int(
        template.get("jee_precise_core_radius", 5)
    )

    dark_threshold = int(
        template.get("jee_precise_dark_threshold", 150)
    )

    filled_threshold = float(
        template.get("jee_precise_filled_threshold", 0.88)
    )

    blank_threshold = float(
        template.get("jee_precise_blank_threshold", 0.80)
    )

    minimum_gap = float(
        template.get("jee_precise_minimum_gap", 0.08)
    )

    translation_delta = float(
        template.get("jee_precise_translation_delta", 14)
    )

    circle_match_delta = float(
        template.get("jee_precise_circle_match_delta", 9)
    )

    detected: Dict[int, Dict[str, Any]] = {}
    calibration: Dict[str, Any] = {}

    for section_index, section in enumerate(
        template.get("mcq_sections", []),
        start=1,
    ):
        options = list(
            section.get(
                "options",
                ["A", "B", "C", "D"],
            )
        )

        x_by_option = {
            option: float(section["option_x"][option])
            for option in options
        }

        y_positions = [
            float(value)
            for value in section["question_y_positions"]
        ]

        start_question = int(
            section["start_question"]
        )

        for row_index, base_y in enumerate(y_positions):
            question_number = (
                start_question
                + row_index
            )

            expected = [
                (x_by_option[option], base_y)
                for option in options
            ]

            circles = _hough_circles(
                gray,
                (
                    min(point[0] for point in expected) - 20,
                    base_y - 20,
                    max(point[0] for point in expected) + 20,
                    base_y + 20,
                ),
                bubble_radius,
            )

            dx, dy, offset_matches = _nearest_offsets(
                circles,
                expected,
                translation_delta,
            )

            translated = [
                (x + dx, y + dy)
                for x, y in expected
            ]

            # Hough circles are used only to estimate row translation.
            #
            # Do NOT use a Hough circle centre as the sampling centre for a
            # filled answer bubble. A heavily filled circle can hide part of
            # the printed ring and shift Hough's centre several pixels toward
            # one side. Q54-B is the exact regression: Hough places it near
            # x=1255.5 while the translated printed centre is x=1251.5.
            #
            # Keep the Hough assignment only for diagnostics, but sample the
            # bubble fill from the translated template centres.
            hough_centres = _assign_unique(
                circles,
                translated,
                circle_match_delta,
            )

            centres = list(translated)

            scores = {
                option: _fill_ratio(
                    gray,
                    centres[index][0],
                    centres[index][1],
                    core_radius,
                    dark_threshold,
                )
                for index, option in enumerate(options)
            }

            ranked = sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            top_option, top_score = ranked[0]

            second_score = (
                ranked[1][1]
                if len(ranked) > 1
                else 0.0
            )

            confidence_gap = float(
                top_score - second_score
            )

            filled_options = [
                option
                for option, score in scores.items()
                if score >= filled_threshold
            ]

            if len(filled_options) >= 2:
                answer = "MULTIPLE"

            elif (
                top_score >= filled_threshold
                and confidence_gap >= minimum_gap
            ):
                answer = top_option

            elif top_score < blank_threshold:
                answer = "BLANK"

            else:
                answer = "UNCERTAIN"

            option_centres = {
                option: [
                    int(round(centres[index][0])),
                    int(round(centres[index][1])),
                ]
                for index, option in enumerate(options)
            }

            detected[question_number] = {
                "answer": answer,
                "scores": {
                    key: round(float(value), 4)
                    for key, value in scores.items()
                },
                "highest_score": round(float(top_score), 4),
                "confidence_gap": round(
                    float(confidence_gap),
                    4,
                ),
                "multiple_options": (
                    filled_options
                    if answer == "MULTIPLE"
                    else []
                ),
                "selected_center": (
                    option_centres.get(top_option)
                    if answer != "BLANK"
                    else None
                ),
                "option_centres": option_centres,
                "reader": "jee_precise_circle_reader_v1",
            }

            calibration[str(question_number)] = {
                "circle_count": len(circles),
                "translation_x": round(float(dx), 3),
                "translation_y": round(float(dy), 3),
                "translation_matches": int(offset_matches),
            }

    return detected, calibration


def _build_numeric_questions(
    section: Dict[str, Any],
    layout: Dict[str, Any],
) -> List[Dict[str, Any]]:
    questions = section.get("questions")

    if questions is not None:
        return list(questions)

    start_question = int(
        section["start_question"]
    )

    digit_offsets = [
        int(value)
        for value in layout["digit_offsets"]
    ]

    decimal_offsets = [
        int(value)
        for value in layout["decimal_offsets"]
    ]

    decimal_after_columns = [
        int(value)
        for value in layout["decimal_after_columns"]
    ]

    digit_values = [
        str(value)
        for value in layout["digit_values"]
    ]

    digit_y_positions = [
        int(value)
        for value in section["digit_y_positions"]
    ]

    result: List[Dict[str, Any]] = []

    for index, base_x_value in enumerate(
        section["question_x_positions"]
    ):
        base_x = int(base_x_value)

        result.append(
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
                        decimal_after_columns,
                    )
                ],
                "sign": {
                    "x": (
                        base_x
                        + int(
                            layout.get(
                                "sign_offset",
                                0,
                            )
                        )
                    ),
                    "y": int(section["sign_y"]),
                },
            }
        )

    return result


def _read_numeric_question(
    gray: np.ndarray,
    question: Dict[str, Any],
    template: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    bubble_radius = int(
        template.get("bubble_radius", 10)
    )

    core_radius = int(
        template.get("jee_precise_core_radius", 5)
    )

    dark_threshold = int(
        template.get("jee_precise_dark_threshold", 150)
    )

    filled_threshold = float(
        template.get(
            "jee_precise_numeric_filled_threshold",
            0.88,
        )
    )

    blank_threshold = float(
        template.get(
            "jee_precise_numeric_blank_threshold",
            0.70,
        )
    )

    minimum_gap = float(
        template.get(
            "jee_precise_numeric_minimum_gap",
            0.10,
        )
    )

    special_threshold = float(
        template.get(
            "jee_precise_numeric_special_threshold",
            0.84,
        )
    )

    translation_delta = float(
        template.get(
            "jee_precise_numeric_translation_delta",
            12,
        )
    )

    circle_match_delta = float(
        template.get(
            "jee_precise_numeric_circle_match_delta",
            10,
        )
    )

    columns = list(
        question.get("columns", [])
    )

    if not columns:
        return {
            "answer": "BLANK",
            "columns": [],
        }, {}

    expected_digit_points: List[Point] = []
    point_meta: List[Tuple[int, str]] = []

    for column_index, column in enumerate(columns):
        x = float(column["x"])

        values = [
            str(value)
            for value in column["values"]
        ]

        y_positions = [
            float(value)
            for value in column["y_positions"]
        ]

        for value, y in zip(values, y_positions):
            expected_digit_points.append((x, y))
            point_meta.append(
                (column_index, value)
            )

    decimal_points = list(
        question.get("decimal_points", [])
    )

    sign = question.get("sign")

    all_x = [
        point[0]
        for point in expected_digit_points
    ]

    all_y = [
        point[1]
        for point in expected_digit_points
    ]

    if decimal_points:
        all_x.extend(
            float(item["x"])
            for item in decimal_points
        )

        all_y.extend(
            float(item["y"])
            for item in decimal_points
        )

    if sign:
        all_x.append(float(sign["x"]))
        all_y.append(float(sign["y"]))

    circles = _hough_circles(
        gray,
        (
            min(all_x) - 20,
            min(all_y) - 20,
            max(all_x) + 20,
            max(all_y) + 20,
        ),
        bubble_radius,
    )

    dx, dy, offset_matches = _nearest_offsets(
        circles,
        expected_digit_points,
        translation_delta,
    )

    translated_digit_points = [
        (x + dx, y + dy)
        for x, y in expected_digit_points
    ]

    # Hough circles are used only to estimate the local translation.
    #
    # Do not sample fill from Hough-assigned centres. A heavily shaded
    # numerical bubble can distort its visible ring and move Hough's centre
    # several pixels. Sampling that shifted point can hit the printed digit/
    # ring of an otherwise blank bubble and create false digits or UNCERTAIN.
    #
    # Estimate dx/dy from the printed grid, then sample the translated
    # template centres themselves.
    hough_digit_points = _assign_unique(
        circles,
        translated_digit_points,
        circle_match_delta,
    )

    column_scores: List[Dict[str, float]] = [
        {}
        for _ in columns
    ]

    column_centres: List[Dict[str, List[int]]] = [
        {}
        for _ in columns
    ]

    for point_index, (
        column_index,
        value,
    ) in enumerate(point_meta):
        centre = translated_digit_points[
            point_index
        ]

        score, sampled_centre = _numeric_local_fill(
            gray,
            centre[0],
            centre[1],
            core_radius,
            dark_threshold,
            search_radius=int(
                template.get(
                    "jee_precise_numeric_local_search_radius",
                    3,
                )
            ),
        )

        column_scores[
            column_index
        ][value] = score

        column_centres[
            column_index
        ][value] = [
            int(round(sampled_centre[0])),
            int(round(sampled_centre[1])),
        ]

    detected_digits: List[str] = []
    column_details: List[Dict[str, Any]] = []

    for column_index, scores in enumerate(
        column_scores
    ):
        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_value, top_score = ranked[0]

        second_score = (
            ranked[1][1]
            if len(ranked) > 1
            else 0.0
        )

        confidence_gap = float(
            top_score - second_score
        )

        relaxed_threshold = float(
            template.get(
                "jee_precise_numeric_relaxed_threshold",
                0.62,
            )
        )

        strong_gap = float(
            template.get(
                "jee_precise_numeric_strong_gap",
                0.10,
            )
        )

        if (
            top_score >= filled_threshold
            and confidence_gap >= minimum_gap
        ):
            detected_value = top_value

        elif (
            top_score >= relaxed_threshold
            and confidence_gap >= strong_gap
        ):
            detected_value = top_value

        elif top_score < blank_threshold:
            detected_value = ""

        else:
            detected_value = "?"

        detected_digits.append(
            detected_value
        )

        column_details.append(
            {
                "column": column_index + 1,
                "value": detected_value,
                "best_score": round(
                    float(top_score),
                    4,
                ),
                "confidence_gap": round(
                    float(confidence_gap),
                    4,
                ),
                "scores": {
                    key: round(float(value), 4)
                    for key, value in scores.items()
                },
                "center": column_centres[
                    column_index
                ].get(top_value),
                "sampling_mode": "translated_template_center",
            }
        )

    decimal_details: List[Dict[str, Any]] = []
    decimal_ranked: List[Tuple[int, float]] = []

    for decimal in decimal_points:
        expected = (
            float(decimal["x"]) + dx,
            float(decimal["y"]) + dy,
        )

        # Keep Hough only for diagnostics; sample the translated template
        # centre so a distorted/filled ring cannot pull the reader sideways.
        hough_centre = _assign_unique(
            circles,
            [expected],
            circle_match_delta,
        )[0]

        centre = expected

        score, sampled_centre = _numeric_local_fill(
            gray,
            centre[0],
            centre[1],
            core_radius,
            dark_threshold,
            search_radius=int(
                template.get(
                    "jee_precise_numeric_local_search_radius",
                    3,
                )
            ),
        )

        centre = sampled_centre

        detail = {
            "after_column": int(
                decimal["after_column"]
            ),
            "score": round(
                float(score),
                4,
            ),
            "center": [
                int(round(centre[0])),
                int(round(centre[1])),
            ],
        }

        decimal_details.append(
            detail
        )

        decimal_ranked.append(
            (
                detail["after_column"],
                float(score),
            )
        )

    decimal_ranked.sort(
        key=lambda item:
        item[1],
        reverse=True,
    )

    selected_decimal = None

    if decimal_ranked:
        best_decimal_column = decimal_ranked[0][0]
        best_decimal_score = float(decimal_ranked[0][1])

        second_decimal_score = float(
            decimal_ranked[1][1]
            if len(decimal_ranked) > 1
            else 0.0
        )

        decimal_gap = (
            best_decimal_score
            - second_decimal_score
        )

        decimal_minimum_gap = float(
            template.get(
                "jee_precise_numeric_decimal_gap",
                0.08,
            )
        )

        if (
            best_decimal_score >= special_threshold
            and decimal_gap >= decimal_minimum_gap
        ):
            selected_decimal = best_decimal_column

    sign_detail = None
    negative = False

    if sign:
        expected = (
            float(sign["x"]) + dx,
            float(sign["y"]) + dy,
        )

        # Keep Hough only for diagnostics; sample the translated template
        # centre for the minus bubble as well.
        hough_centre = _assign_unique(
            circles,
            [expected],
            circle_match_delta,
        )[0]

        centre = expected

        score, sampled_centre = _numeric_local_fill(
            gray,
            centre[0],
            centre[1],
            core_radius,
            dark_threshold,
            search_radius=int(
                template.get(
                    "jee_precise_numeric_local_search_radius",
                    3,
                )
            ),
        )

        centre = sampled_centre

        negative = (
            score >= special_threshold
        )

        sign_detail = {
            "value": "-" if negative else "",
            "score": round(
                float(score),
                4,
            ),
            "center": [
                int(round(centre[0])),
                int(round(centre[1])),
            ],
        }

    active_indices = [
        index
        for index, value in enumerate(
            detected_digits
        )
        if value != ""
    ]

    if not active_indices:
        answer = "BLANK"

    else:
        first = min(active_indices)
        last = max(active_indices)

        used_digits = detected_digits[
            first:last + 1
        ]

        if any(
            value in ("", "?")
            for value in used_digits
        ):
            answer = "UNCERTAIN"

        else:
            answer = "".join(
                used_digits
            )

            if selected_decimal is not None:
                after_column = (
                    selected_decimal
                )

                insertion = (
                    after_column
                    - first
                )

                if (
                    insertion <= 0
                    or
                    insertion >= len(answer)
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
                and
                answer != "UNCERTAIN"
            ):
                answer = (
                    "-"
                    + answer
                )

    return {
        "answer": answer,
        "columns": column_details,
        "decimal_points": decimal_details,
        "selected_decimal": selected_decimal,
        "sign": sign_detail,
        "reader": "jee_precise_circle_reader_v2",
    }, {
        "circle_count": len(circles),
        "translation_x": round(float(dx), 3),
        "translation_y": round(float(dy), 3),
        "translation_matches": int(offset_matches),
    }


def scan_jee_numerical_precise(
    corrected_image: np.ndarray,
    template: Dict[str, Any],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    gray = _gray(corrected_image)

    layout = template.get(
        "numerical_layout",
        {},
    )

    detected: Dict[int, Dict[str, Any]] = {}
    calibration: Dict[str, Any] = {}

    for section in template.get(
        "numerical_sections",
        [],
    ):
        questions = _build_numeric_questions(
            section,
            layout,
        )

        for question in questions:
            question_number = int(
                question["question"]
            )

            calibrated_question, grid_debug = (
                _calibrate_numerical_question(
                    gray,
                    question,
                    template,
                )
            )

            record, debug = (
                _read_numeric_question(
                    gray,
                    calibrated_question,
                    template,
                )
            )

            debug["grid_cluster"] = grid_debug

            detected[
                question_number
            ] = record

            calibration[
                str(question_number)
            ] = debug

    return detected, calibration


def scan_jee_answers_precise(
    corrected_image: np.ndarray,
    template: Dict[str, Any],
) -> Dict[str, Any]:
    mcq, mcq_debug = (
        scan_jee_mcq_precise(
            corrected_image,
            template,
        )
    )

    numerical, numerical_debug = (
        scan_jee_numerical_precise(
            corrected_image,
            template,
        )
    )

    return {
        "mcq": mcq,
        "numerical": numerical,
        "_calibration": {
            "mcq": mcq_debug,
            "numerical": numerical_debug,
        },
    }
