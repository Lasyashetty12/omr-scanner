from __future__ import annotations

import math

import cv2
import numpy as np


PROFILE = "shared_row_ring_alignment_v10_22"


def _as_gray(image):
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _ring_contrast(gray, center_x, center_y, ring_radius):
    """
    Score how well one candidate center is aligned with a printed bubble ring.

    The bubble core is deliberately ignored so the same geometry score works
    for both empty and filled bubbles.
    """
    radius = max(6.0, float(ring_radius))
    outer_radius = max(radius + 3.0, radius * 1.36)

    height, width = gray.shape[:2]

    x1 = max(0, int(math.floor(float(center_x) - outer_radius - 1.0)))
    y1 = max(0, int(math.floor(float(center_y) - outer_radius - 1.0)))
    x2 = min(width, int(math.ceil(float(center_x) + outer_radius + 2.0)))
    y2 = min(height, int(math.ceil(float(center_y) + outer_radius + 2.0)))

    if x2 - x1 < 12 or y2 - y1 < 12:
        return None

    patch = gray[y1:y2, x1:x2]

    yy, xx = np.ogrid[y1:y2, x1:x2]
    distance = np.sqrt(
        (xx - float(center_x)) ** 2
        + (yy - float(center_y)) ** 2
    )

    # Manchester response bubbles are close to bubble_radius on the canonical
    # image. Use the printed annulus, not the center mark, as geometry evidence.
    ring_mask = (
        (distance >= radius * 0.58)
        & (distance <= radius * 1.02)
    )
    paper_mask = (
        (distance >= radius * 1.10)
        & (distance <= radius * 1.34)
    )

    if (
        np.count_nonzero(ring_mask) < 20
        or np.count_nonzero(paper_mask) < 16
    ):
        return None

    ring_level = float(np.mean(patch[ring_mask]))
    paper_level = float(np.percentile(patch[paper_mask], 70.0))

    return float(
        np.clip(
            paper_level - ring_level,
            -40.0,
            160.0,
        )
    )


def _row_score(
    gray,
    option_map,
    shift_x,
    shift_y,
    ring_radius,
):
    option_scores = {}

    for option, (center_x, center_y) in option_map.items():
        score = _ring_contrast(
            gray,
            float(center_x) + float(shift_x),
            float(center_y) + float(shift_y),
            ring_radius,
        )

        if score is not None:
            option_scores[str(option)] = float(score)

    if len(option_scores) < 3:
        return None

    values = list(option_scores.values())
    median_score = float(np.median(values))
    mean_score = float(np.mean(values))

    # Prefer the existing fitted position if two shifts are effectively tied.
    combined = (
        0.75 * median_score
        + 0.25 * mean_score
        - 0.18 * abs(float(shift_x))
        - 0.28 * abs(float(shift_y))
    )

    return {
        "score": float(combined),
        "support": int(
            sum(value >= 3.0 for value in values)
        ),
        "option_scores": option_scores,
    }


def _shared_shift_bounds(
    current_map,
    calibrated_map,
    template,
):
    search_x = max(
        0,
        min(
            4,
            int(
                template.get(
                    "grid_row_refine_search_x",
                    3,
                )
            ),
        ),
    )

    search_y = max(
        0,
        min(
            4,
            int(
                template.get(
                    "grid_row_refine_search_y",
                    3,
                )
            ),
        ),
    )

    # Preserve the existing safe geometry envelope. In particular, do NOT
    # restore the old KCET 10/16-pixel Y window that could reach another row.
    max_dx_from_calibrated = float(
        template.get(
            "grid_max_final_dx_from_input",
            8.0,
        )
    )
    max_dy_from_calibrated = float(
        template.get(
            "grid_max_final_dy_from_input",
            10.0,
        )
    )

    min_dx, max_dx = -search_x, search_x
    min_dy, max_dy = -search_y, search_y

    for option, current_center in current_map.items():
        if option not in calibrated_map:
            continue

        current_x = float(current_center[0])
        current_y = float(current_center[1])

        calibrated_x = float(
            calibrated_map[option][0]
        )
        calibrated_y = float(
            calibrated_map[option][1]
        )

        min_dx = max(
            min_dx,
            int(
                math.ceil(
                    calibrated_x
                    - max_dx_from_calibrated
                    - current_x
                )
            ),
        )
        max_dx = min(
            max_dx,
            int(
                math.floor(
                    calibrated_x
                    + max_dx_from_calibrated
                    - current_x
                )
            ),
        )
        min_dy = max(
            min_dy,
            int(
                math.ceil(
                    calibrated_y
                    - max_dy_from_calibrated
                    - current_y
                )
            ),
        )
        max_dy = min(
            max_dy,
            int(
                math.floor(
                    calibrated_y
                    + max_dy_from_calibrated
                    - current_y
                )
            ),
        )

    return (
        int(min_dx),
        int(max_dx),
        int(min_dy),
        int(max_dy),
    )


def refine_fitted_grid_to_printed_rings(
    gray,
    fitted_coordinates,
    calibrated_coordinates,
    template,
):
    """
    Apply one tiny shared (dx, dy) correction per question row.

    Safety rules:
      * A/B/C/D move together.
      * Search is +/-3 px by default.
      * The existing template/grid limits are never exceeded.
      * At least three physical bubble rings must improve together.
      * NEET/KCET only; JEE is untouched.
    """
    gray = _as_gray(gray)

    refined = {
        question: {
            option: (
                float(center[0]),
                float(center[1]),
            )
            for option, center in option_map.items()
        }
        for question, option_map
        in fitted_coordinates.items()
    }

    exam_name = str(
        template.get(
            "exam_name",
            "",
        )
    ).strip().upper()

    if exam_name not in ("NEET", "KCET"):
        return refined, {}

    questions_per_column = max(
        1,
        int(
            template.get(
                "questions_per_column",
                60,
            )
        ),
    )

    ring_radius = float(
        template.get(
            "bubble_radius",
            11,
        )
    )

    minimum_improvement = float(
        template.get(
            "grid_row_refine_min_improvement",
            2.0,
        )
    )

    minimum_option_improvement = float(
        template.get(
            "grid_row_refine_min_option_improvement",
            0.75,
        )
    )

    diagnostics = {}

    for question, option_map in refined.items():
        try:
            question_no = int(question)
        except (TypeError, ValueError):
            continue

        calibrated_map = calibrated_coordinates.get(
            question,
            calibrated_coordinates.get(
                question_no,
                calibrated_coordinates.get(
                    str(question_no),
                    {},
                ),
            ),
        )

        if (
            not isinstance(calibrated_map, dict)
            or len(option_map) < 3
        ):
            continue

        (
            min_dx,
            max_dx,
            min_dy,
            max_dy,
        ) = _shared_shift_bounds(
            option_map,
            calibrated_map,
            template,
        )

        if min_dx > max_dx or min_dy > max_dy:
            continue

        base = _row_score(
            gray,
            option_map,
            0,
            0,
            ring_radius,
        )

        if base is None:
            continue

        best = {
            **base,
            "dx": 0,
            "dy": 0,
        }

        for shift_y in range(
            min_dy,
            max_dy + 1,
        ):
            for shift_x in range(
                min_dx,
                max_dx + 1,
            ):
                if shift_x == 0 and shift_y == 0:
                    continue

                candidate = _row_score(
                    gray,
                    option_map,
                    shift_x,
                    shift_y,
                    ring_radius,
                )

                if candidate is None:
                    continue

                candidate_key = (
                    float(candidate["score"]),
                    -(
                        abs(int(shift_x))
                        + abs(int(shift_y))
                    ),
                    -abs(int(shift_y)),
                    -abs(int(shift_x)),
                )

                best_key = (
                    float(best["score"]),
                    -(
                        abs(int(best["dx"]))
                        + abs(int(best["dy"]))
                    ),
                    -abs(int(best["dy"])),
                    -abs(int(best["dx"])),
                )

                if candidate_key > best_key:
                    best = {
                        **candidate,
                        "dx": int(shift_x),
                        "dy": int(shift_y),
                    }

        improvement = float(
            best["score"] - base["score"]
        )

        common_options = (
            set(base["option_scores"])
            & set(best["option_scores"])
        )

        improved_options = sum(
            best["option_scores"][option]
            >= (
                base["option_scores"][option]
                + minimum_option_improvement
            )
            for option in common_options
        )

        should_apply = bool(
            (
                best["dx"] != 0
                or best["dy"] != 0
            )
            and best["support"] >= 3
            and improved_options >= 3
            and improvement >= minimum_improvement
        )

        if not should_apply:
            continue

        shift_x = int(best["dx"])
        shift_y = int(best["dy"])

        for option, center in list(
            option_map.items()
        ):
            option_map[option] = (
                float(center[0]) + shift_x,
                float(center[1]) + shift_y,
            )

        column_index = max(
            0,
            (
                question_no - 1
            )
            // questions_per_column,
        )

        column_debug = diagnostics.setdefault(
            column_index,
            {
                "profile": PROFILE,
                "applied_rows": 0,
                "rows": [],
            },
        )

        column_debug["applied_rows"] += 1
        column_debug["rows"].append(
            {
                "question": question_no,
                "dx": shift_x,
                "dy": shift_y,
                "before_score": round(
                    float(base["score"]),
                    3,
                ),
                "after_score": round(
                    float(best["score"]),
                    3,
                ),
                "improvement": round(
                    improvement,
                    3,
                ),
                "improved_options": int(
                    improved_options
                ),
            }
        )

    return refined, diagnostics
