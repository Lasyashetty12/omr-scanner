from __future__ import annotations

import cv2
import numpy as np

from .grid_detector import detect_circle_candidates


PROFILE = "cv_detected_bubble_lattice_v10_23"


def _gray(image):
    return (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )


def _qrange(column_index, qpc):
    start = column_index * qpc + 1
    return start, start + qpc


def _roi(seed_coordinates, start_q, end_q):
    xs, ys = [], []
    for q in range(start_q, end_q):
        option_map = seed_coordinates.get(q, seed_coordinates.get(str(q), {}))
        for x, y in option_map.values():
            xs.append(float(x))
            ys.append(float(y))
    if not xs or not ys:
        return None
    return (
        min(xs) - 36.0,
        min(ys) - 24.0,
        max(xs) + 36.0,
        max(ys) + 24.0,
    )


def _kmeans_1d(values, k=4):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < k:
        return None

    centers = np.percentile(
        values,
        np.linspace(12.5, 87.5, k),
    ).astype(np.float32)

    for _ in range(24):
        labels = np.argmin(
            np.abs(values[:, None] - centers[None, :]),
            axis=1,
        )
        updated = centers.copy()
        for i in range(k):
            group = values[labels == i]
            if group.size:
                updated[i] = float(np.median(group))
        updated = np.sort(updated)
        if np.max(np.abs(updated - centers)) < 0.05:
            centers = updated
            break
        centers = updated

    return np.sort(centers)


def _cluster_rows(candidates, tolerance):
    rows = []
    for candidate in sorted(candidates, key=lambda item: float(item["y"])):
        if not rows:
            rows.append([candidate])
            continue

        current_y = float(
            np.median([float(item["y"]) for item in rows[-1]])
        )
        if float(candidate["y"]) - current_y <= tolerance:
            rows[-1].append(candidate)
        else:
            rows.append([candidate])
    return rows


def _pick_cells(row_candidates, lane_centers, lane_tolerance):
    cells = {}
    for candidate in row_candidates:
        x = float(candidate["x"])
        distances = np.abs(lane_centers - x)
        lane = int(np.argmin(distances))
        if float(distances[lane]) > lane_tolerance:
            continue

        old = cells.get(lane)
        if old is None:
            cells[lane] = candidate
            continue

        old_key = (
            float(old.get("score", 0.0)),
            -abs(float(old["x"]) - float(lane_centers[lane])),
        )
        new_key = (
            float(candidate.get("score", 0.0)),
            -abs(x - float(lane_centers[lane])),
        )
        if new_key > old_key:
            cells[lane] = candidate

    return cells


def _expected_y(seed_coordinates, start_q, qpc, first_option):
    values = []
    for row in range(qpc):
        q = start_q + row
        option_map = seed_coordinates.get(q, seed_coordinates.get(str(q), {}))
        if first_option in option_map:
            values.append(float(option_map[first_option][1]))
        elif option_map:
            values.append(
                float(np.median([float(center[1]) for center in option_map.values()]))
            )
        else:
            values.append(np.nan)
    return np.asarray(values, dtype=np.float32)


def _map_rows(detected_rows, expected_y, qpc):
    """
    When all physical rows are detected, CV row order is the authority.
    Template Y is used only if a few rows are missing/extra and we need to
    decide which question number belongs to which detected physical row.
    """
    detected_rows = sorted(detected_rows, key=lambda row: float(row["y"]))

    if len(detected_rows) == qpc:
        return {index: detected_rows[index] for index in range(qpc)}

    finite = expected_y[np.isfinite(expected_y)]
    if finite.size < max(4, qpc // 2):
        return {}

    spacing = float(np.median(np.diff(finite)))
    if spacing <= 0:
        return {}

    rough_offsets = []
    for row in detected_rows:
        y = float(row["y"])
        nearest = int(np.argmin(np.abs(expected_y - y)))
        if np.isfinite(expected_y[nearest]):
            rough_offsets.append(y - float(expected_y[nearest]))

    offset = float(np.median(rough_offsets)) if rough_offsets else 0.0
    targets = expected_y + offset
    max_distance = max(7.0, min(14.0, spacing * 0.48))

    possible = []
    for detected_index, row in enumerate(detected_rows):
        y = float(row["y"])
        for row_index in range(qpc):
            target = float(targets[row_index])
            if not np.isfinite(target):
                continue
            distance = abs(y - target)
            if distance <= max_distance:
                possible.append(
                    (
                        distance,
                        -int(row.get("support", 0)),
                        detected_index,
                        row_index,
                    )
                )

    possible.sort()
    used_detected, used_rows, mapped = set(), set(), {}

    for _, _, detected_index, row_index in possible:
        if detected_index in used_detected or row_index in used_rows:
            continue
        used_detected.add(detected_index)
        used_rows.add(row_index)
        mapped[row_index] = detected_rows[detected_index]

    return mapped


def _row_y_series(mapped_rows, expected_y, qpc):
    known = sorted(mapped_rows)
    if not known:
        return np.asarray(expected_y, dtype=np.float32)

    known_y = np.asarray(
        [float(mapped_rows[index]["y"]) for index in known],
        dtype=np.float32,
    )

    if len(known) >= 2:
        spacing = float(
            np.median(
                np.diff(known_y)
                / np.diff(np.asarray(known, dtype=np.float32))
            )
        )
    else:
        finite = expected_y[np.isfinite(expected_y)]
        spacing = (
            float(np.median(np.diff(finite)))
            if finite.size >= 2
            else 28.0
        )

    result = np.empty(qpc, dtype=np.float32)

    for row in range(qpc):
        if row in mapped_rows:
            result[row] = float(mapped_rows[row]["y"])
            continue

        if row < known[0]:
            result[row] = known_y[0] - (known[0] - row) * spacing
            continue

        if row > known[-1]:
            result[row] = known_y[-1] + (row - known[-1]) * spacing
            continue

        right_pos = int(np.searchsorted(known, row))
        left_index = known[right_pos - 1]
        right_index = known[right_pos]
        left_y = float(mapped_rows[left_index]["y"])
        right_y = float(mapped_rows[right_index]["y"])
        alpha = (row - left_index) / float(right_index - left_index)
        result[row] = left_y + alpha * (right_y - left_y)

    return result


def _lane_track(samples, fallback_x):
    if len(samples) < 6:
        return lambda _y: float(fallback_x)

    y = np.asarray([float(item[0]) for item in samples], dtype=np.float64)
    x = np.asarray([float(item[1]) for item in samples], dtype=np.float64)

    y0 = float(np.median(y))
    scale = max(1.0, float(np.ptp(y)))
    yn = (y - y0) / scale
    degree = 2 if len(samples) >= 20 else 1
    mask = np.ones(len(samples), dtype=bool)
    coeff = None

    for _ in range(3):
        if np.count_nonzero(mask) < degree + 2:
            break
        coeff = np.polyfit(yn[mask], x[mask], degree)
        residual = np.abs(x - np.polyval(coeff, yn))
        median_residual = float(np.median(residual[mask]))
        threshold = max(2.5, min(7.0, median_residual * 3.0 + 1.5))
        new_mask = residual <= threshold
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    if coeff is None:
        center = float(np.median(x))
        return lambda _y: center

    def predict(y_value):
        return float(np.polyval(coeff, (float(y_value) - y0) / scale))

    return predict


def _fit_column(gray, fallback, seed, template, column_index):
    options = list(template.get("options", ["A", "B", "C", "D"]))
    if len(options) != 4:
        return None, {"profile": PROFILE, "status": "fallback_non_four_option"}

    qpc = int(template.get("questions_per_column", 60))
    start_q, end_q = _qrange(column_index, qpc)
    roi = _roi(seed, start_q, end_q)
    if roi is None:
        return None, {"profile": PROFILE, "status": "fallback_missing_roi"}

    candidates = detect_circle_candidates(gray, roi)
    minimum_candidates = max(60, int(qpc * len(options) * 0.35))
    if len(candidates) < minimum_candidates:
        return None, {
            "profile": PROFILE,
            "status": "fallback_too_few_candidates",
            "candidate_count": len(candidates),
        }

    lane_centers = _kmeans_1d([float(item["x"]) for item in candidates], 4)
    if lane_centers is None:
        return None, {"profile": PROFILE, "status": "fallback_lane_fit_failed"}

    lane_gaps = np.diff(lane_centers)
    if (
        lane_gaps.size != 3
        or float(np.min(lane_gaps)) < 20.0
        or float(np.max(lane_gaps)) > 80.0
    ):
        return None, {
            "profile": PROFILE,
            "status": "fallback_bad_lane_spacing",
            "lane_centers": [round(float(value), 2) for value in lane_centers],
        }

    lane_spacing = float(np.median(lane_gaps))
    lane_tolerance = max(9.0, min(18.0, lane_spacing * 0.40))

    lane_candidates = []
    for candidate in candidates:
        distances = np.abs(lane_centers - float(candidate["x"]))
        lane = int(np.argmin(distances))
        if float(distances[lane]) > lane_tolerance:
            continue
        item = dict(candidate)
        item["_lane"] = lane
        lane_candidates.append(item)

    expected_y = _expected_y(seed, start_q, qpc, options[0])
    finite = expected_y[np.isfinite(expected_y)]
    expected_spacing = (
        float(np.median(np.diff(finite)))
        if finite.size >= 2
        else 28.0
    )
    row_tolerance = max(4.0, min(7.0, expected_spacing * 0.23))

    detected_rows = []
    for cluster in _cluster_rows(lane_candidates, row_tolerance):
        cells = _pick_cells(cluster, lane_centers, lane_tolerance)
        if len(cells) < 2:
            continue
        detected_rows.append(
            {
                "y": float(
                    np.median([float(candidate["y"]) for candidate in cells.values()])
                ),
                "support": len(cells),
                "cells": cells,
            }
        )

    mapped_rows = _map_rows(detected_rows, expected_y, qpc)
    detected_row_count = len(mapped_rows)
    direct_cell_count = sum(len(row["cells"]) for row in mapped_rows.values())

    if (
        detected_row_count < max(42, int(round(qpc * 0.78)))
        or direct_cell_count < int(round(qpc * len(options) * 0.62))
    ):
        return None, {
            "profile": PROFILE,
            "status": "fallback_low_cv_coverage",
            "candidate_count": len(candidates),
            "detected_row_count": detected_row_count,
            "direct_cell_count": direct_cell_count,
        }

    row_y = _row_y_series(mapped_rows, expected_y, qpc)

    lane_samples = {lane: [] for lane in range(4)}
    for row in mapped_rows.values():
        for lane, candidate in row["cells"].items():
            lane_samples[lane].append((float(row["y"]), float(candidate["x"])))

    lane_predictors = {}
    for lane, option in enumerate(options):
        seed_x = []
        for row in range(qpc):
            q = start_q + row
            option_map = seed.get(q, seed.get(str(q), {}))
            if option in option_map:
                seed_x.append(float(option_map[option][0]))
        fallback_x = (
            float(np.median(seed_x))
            if seed_x
            else float(lane_centers[lane])
        )
        lane_predictors[lane] = _lane_track(lane_samples[lane], fallback_x)

    fitted = {}
    direct_used = 0

    for row in range(qpc):
        q = start_q + row
        physical_y = float(row_y[row])
        fitted[q] = {}
        detected_row = mapped_rows.get(row)

        for lane, option in enumerate(options):
            candidate = (
                detected_row["cells"].get(lane)
                if detected_row is not None
                else None
            )
            if candidate is not None:
                x = float(candidate["x"])
                direct_used += 1
            else:
                x = float(lane_predictors[lane](physical_y))

            # All A/B/C/D in one question share the CV-detected physical row Y.
            fitted[q][option] = (x, physical_y)

    spacing_values = np.diff(row_y)
    spacing_values = spacing_values[spacing_values > 0]

    return fitted, {
        "profile": PROFILE,
        "status": "cv_lattice_authority",
        "candidate_count": len(candidates),
        "detected_row_count": detected_row_count,
        "expected_row_count": qpc,
        "direct_cell_count": direct_used,
        "total_cell_count": qpc * len(options),
        "direct_cell_coverage": round(
            direct_used / float(qpc * len(options)),
            4,
        ),
        "lane_centers": [round(float(value), 2) for value in lane_centers],
        "median_row_spacing": round(
            float(np.median(spacing_values)) if spacing_values.size else 0.0,
            3,
        ),
        "roi": [round(float(value), 2) for value in roi],
        "template_used_for": [
            "broad response-block ROI",
            "question count/order",
            "fallback numbering only",
        ],
        "final_geometry_source": "CV detected physical bubble rings",
    }


def refine_fitted_grid_to_printed_rings(
    gray,
    fitted_coordinates,
    calibrated_coordinates,
    template,
):
    """
    CV-authoritative NEET/KCET bubble positioning.

    Unlike v10.22, this does not search only +/-3 px around template centers.
    It detects the physical bubble lattice and uses those detected centers.
    Template coordinates are only a coarse ROI/numbering prior and fallback.
    """
    gray = _gray(gray)

    refined = {
        int(question): {
            str(option): (float(center[0]), float(center[1]))
            for option, center in option_map.items()
        }
        for question, option_map in fitted_coordinates.items()
        if isinstance(option_map, dict)
    }

    exam_name = str(template.get("exam_name", "")).strip().upper()
    if exam_name not in ("NEET", "KCET"):
        return refined, {}

    diagnostics = {}

    for column_index in range(len(template.get("columns", []))):
        cv_fitted, debug = _fit_column(
            gray,
            refined,
            calibrated_coordinates,
            template,
            column_index,
        )
        diagnostics[column_index] = debug
        if cv_fitted is not None:
            refined.update(cv_fitted)

    return refined, diagnostics
