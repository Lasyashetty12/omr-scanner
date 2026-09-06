from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from jee_reader import (
    scan_jee_mcq_sections_robust as _legacy_scan_jee_mcq_sections_robust,
)
from ml_omr.grid_detector import detect_circle_candidates


PROFILE = "jee_cv_physical_mcq_lattice_v10_25"


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _best_axis_offset(
    values: List[float],
    seeds: List[float],
    search_radius: float,
    match_tolerance: float,
) -> Tuple[float, int, float]:
    """Find the translation with the strongest physical-circle agreement."""
    if not values or not seeds:
        return 0.0, 0, 999.0

    values_arr = np.asarray(values, dtype=np.float32)
    seeds_arr = np.asarray(seeds, dtype=np.float32)

    best = (0.0, -1, float("inf"))

    for offset in np.arange(
        -float(search_radius),
        float(search_radius) + 0.25,
        0.5,
        dtype=np.float32,
    ):
        distances = np.min(
            np.abs(
                values_arr[:, None]
                - (seeds_arr[None, :] + float(offset))
            ),
            axis=1,
        )

        accepted = distances <= float(match_tolerance)
        support = int(np.count_nonzero(accepted))
        median_error = (
            float(np.median(distances[accepted]))
            if support
            else float("inf")
        )

        candidate_key = (
            support,
            -median_error,
            -abs(float(offset)),
        )
        best_key = (
            best[1],
            -best[2],
            -abs(best[0]),
        )

        if candidate_key > best_key:
            best = (
                float(offset),
                support,
                median_error,
            )

    return best


def _fit_lane_track(
    samples: List[Tuple[float, float]],
    fallback_x: float,
):
    """Fit X as a function of Y for the few cells whose ring is not detected."""
    if len(samples) < 3:
        return lambda _y: float(fallback_x)

    y = np.asarray(
        [float(item[0]) for item in samples],
        dtype=np.float64,
    )
    x = np.asarray(
        [float(item[1]) for item in samples],
        dtype=np.float64,
    )

    y0 = float(np.median(y))
    scale = max(1.0, float(np.ptp(y)))
    yn = (y - y0) / scale
    degree = 2 if len(samples) >= 12 else 1

    mask = np.ones(len(samples), dtype=bool)
    coefficients = None

    for _ in range(3):
        if np.count_nonzero(mask) < degree + 2:
            break

        coefficients = np.polyfit(
            yn[mask],
            x[mask],
            degree,
        )

        residual = np.abs(
            x
            - np.polyval(
                coefficients,
                yn,
            )
        )

        active = residual[mask]
        median_residual = (
            float(np.median(active))
            if active.size
            else 0.0
        )

        threshold = max(
            2.0,
            min(
                6.0,
                median_residual * 3.0 + 1.0,
            ),
        )

        new_mask = residual <= threshold

        if np.array_equal(
            new_mask,
            mask,
        ):
            break

        mask = new_mask

    if coefficients is None:
        centre = float(np.median(x))
        return lambda _y: centre

    def predict(y_value: float) -> float:
        return float(
            np.polyval(
                coefficients,
                (float(y_value) - y0) / scale,
            )
        )

    return predict


def _fit_section(
    gray: np.ndarray,
    section: Dict[str, Any],
    template: Dict[str, Any],
):
    options = list(
        section.get(
            "options",
            ["A", "B", "C", "D"],
        )
    )

    total_questions = int(
        section["total_questions"]
    )

    expected_y = [
        float(value)
        for value in section["question_y_positions"]
    ]

    expected_x = [
        float(section["option_x"][option])
        for option in options
    ]

    if (
        len(options) != 4
        or len(expected_y) != total_questions
    ):
        return None, {
            "profile": PROFILE,
            "status": "fallback_bad_section_layout",
        }

    margin_x = float(
        template.get(
            "jee_cv_mcq_roi_margin_x",
            32.0,
        )
    )
    margin_y = float(
        template.get(
            "jee_cv_mcq_roi_margin_y",
            22.0,
        )
    )

    roi = (
        min(expected_x) - margin_x,
        min(expected_y) - margin_y,
        max(expected_x) + margin_x,
        max(expected_y) + margin_y,
    )

    # Same physical-circle detector used by the NEET/KCET grid path.
    candidates = detect_circle_candidates(
        gray,
        roi,
    )

    minimum_candidates = max(
        18,
        int(
            round(
                total_questions
                * len(options)
                * 0.42
            )
        ),
    )

    if len(candidates) < minimum_candidates:
        return None, {
            "profile": PROFILE,
            "status": "fallback_too_few_physical_circles",
            "candidate_count": len(candidates),
        }

    x_offset, x_support, x_error = _best_axis_offset(
        [
            float(candidate["x"])
            for candidate in candidates
        ],
        expected_x,
        float(
            template.get(
                "jee_cv_mcq_x_search",
                28.0,
            )
        ),
        float(
            template.get(
                "jee_cv_mcq_x_seed_tolerance",
                7.0,
            )
        ),
    )

    y_offset, y_support, y_error = _best_axis_offset(
        [
            float(candidate["y"])
            for candidate in candidates
        ],
        expected_y,
        float(
            template.get(
                "jee_cv_mcq_y_search",
                24.0,
            )
        ),
        float(
            template.get(
                "jee_cv_mcq_y_seed_tolerance",
                8.0,
            )
        ),
    )

    seeded_x = (
        np.asarray(
            expected_x,
            dtype=np.float32,
        )
        + x_offset
    )

    seeded_y = (
        np.asarray(
            expected_y,
            dtype=np.float32,
        )
        + y_offset
    )

    tolerance_x = float(
        template.get(
            "jee_cv_mcq_cell_tolerance_x",
            10.0,
        )
    )
    tolerance_y = float(
        template.get(
            "jee_cv_mcq_cell_tolerance_y",
            10.0,
        )
    )

    cells = {}

    for candidate in candidates:
        cx = float(candidate["x"])
        cy = float(candidate["y"])

        lane = int(
            np.argmin(
                np.abs(
                    seeded_x - cx
                )
            )
        )
        row = int(
            np.argmin(
                np.abs(
                    seeded_y - cy
                )
            )
        )

        dx = abs(
            cx - float(seeded_x[lane])
        )
        dy = abs(
            cy - float(seeded_y[row])
        )

        if (
            dx > tolerance_x
            or dy > tolerance_y
        ):
            continue

        key = (row, lane)

        quality = (
            float(
                candidate.get(
                    "score",
                    0.0,
                )
            )
            - 0.05 * (dx + dy)
        )

        previous = cells.get(key)

        if (
            previous is None
            or quality
            > float(previous["_quality"])
        ):
            cells[key] = {
                **candidate,
                "_quality": quality,
            }

    covered_rows = {
        row
        for row, _lane in cells
    }

    minimum_rows = max(
        7,
        int(
            round(
                total_questions
                * 0.75
            )
        ),
    )
    minimum_cells = max(
        22,
        int(
            round(
                total_questions
                * len(options)
                * 0.55
            )
        ),
    )

    if (
        len(covered_rows) < minimum_rows
        or len(cells) < minimum_cells
    ):
        return None, {
            "profile": PROFILE,
            "status": "fallback_low_cv_coverage",
            "candidate_count": len(candidates),
            "direct_cell_count": len(cells),
            "covered_row_count": len(covered_rows),
            "x_offset": round(x_offset, 3),
            "y_offset": round(y_offset, 3),
        }

    row_samples = {
        row: []
        for row in range(total_questions)
    }
    lane_samples = {
        lane: []
        for lane in range(len(options))
    }

    for (row, lane), candidate in cells.items():
        cx = float(candidate["x"])
        cy = float(candidate["y"])

        row_samples[row].append(cy)
        lane_samples[lane].append(
            (cy, cx)
        )

    detected_row_y = {
        row: float(np.median(values))
        for row, values in row_samples.items()
        if values
    }

    known_rows = sorted(
        detected_row_y
    )

    row_y = seeded_y.copy()

    if len(known_rows) >= 2:
        known_indices = np.asarray(
            known_rows,
            dtype=np.float32,
        )
        known_values = np.asarray(
            [
                detected_row_y[row]
                for row in known_rows
            ],
            dtype=np.float32,
        )

        row_y = np.interp(
            np.arange(
                total_questions,
                dtype=np.float32,
            ),
            known_indices,
            known_values,
        ).astype(np.float32)

        observed_steps = (
            np.diff(known_values)
            /
            np.maximum(
                1.0,
                np.diff(known_indices),
            )
        )

        spacing = (
            float(np.median(observed_steps))
            if observed_steps.size
            else float(
                np.median(
                    np.diff(seeded_y)
                )
            )
        )

        first_row = known_rows[0]
        last_row = known_rows[-1]

        for row in range(0, first_row):
            row_y[row] = (
                known_values[0]
                - (first_row - row) * spacing
            )

        for row in range(
            last_row + 1,
            total_questions,
        ):
            row_y[row] = (
                known_values[-1]
                + (row - last_row) * spacing
            )

    lane_tracks = {
        lane: _fit_lane_track(
            lane_samples[lane],
            float(seeded_x[lane]),
        )
        for lane in range(len(options))
    }

    start_question = int(
        section["start_question"]
    )
    output = {}
    direct_count = 0

    for row in range(total_questions):
        question_number = (
            start_question + row
        )
        physical_row_y = float(
            row_y[row]
        )

        option_centres = {}
        centre_sources = {}

        for lane, option in enumerate(options):
            direct = cells.get(
                (row, lane)
            )

            if direct is not None:
                cx = float(direct["x"])
                cy = float(direct["y"])
                source = "physical_circle"
                direct_count += 1
            else:
                cx = float(
                    lane_tracks[lane](
                        physical_row_y
                    )
                )
                cy = physical_row_y
                source = "cv_lattice_interpolation"

            option_centres[option] = [
                int(round(cx)),
                int(round(cy)),
            ]
            centre_sources[option] = source

        # scanner.py uses option_centres only; hybrid_reader performs
        # the actual filled/blank/multiple decision afterwards.
        output[question_number] = {
            "answer": "UNCERTAIN",
            "scores": {},
            "reader": PROFILE,
            "grid_calibrated": True,
            "option_centres": option_centres,
            "centre_sources": centre_sources,
        }

    total_cells = (
        total_questions
        * len(options)
    )

    debug = {
        "profile": PROFILE,
        "status": "cv_physical_circle_authority",
        "candidate_count": len(candidates),
        "direct_cell_count": direct_count,
        "total_cell_count": total_cells,
        "direct_cell_coverage": round(
            direct_count / float(total_cells),
            4,
        ),
        "covered_row_count": len(covered_rows),
        "total_row_count": total_questions,
        "x_offset": round(x_offset, 3),
        "y_offset": round(y_offset, 3),
        "x_support": x_support,
        "y_support": y_support,
        "x_median_error": round(x_error, 3),
        "y_median_error": round(y_error, 3),
        "final_geometry_source":
            "detected printed bubble centres",
        "template_role":
            "coarse ROI and question/option labels only",
    }

    return output, debug


def scan_jee_mcq_sections_cv_robust(
    corrected_image: np.ndarray,
    template: Dict[str, Any],
):
    """
    Detect JEE MCQ bubble centres from the physical printed circles.

    The template is no longer the final centre authority. It is used only to
    locate each MCQ block and label rows/options. Direct CV centres are used
    cell-by-cell; missing individual rings are interpolated from neighbouring
    detected rows/lanes. If a section has insufficient CV evidence, the
    previous JEE reader is used for that section only.
    """
    gray = _gray(
        corrected_image
    )

    output = {}
    debug = {
        "profile": PROFILE,
        "sections": {},
    }

    legacy_records = None
    legacy_debug = None

    for section_index, section in enumerate(
        template.get(
            "mcq_sections",
            [],
        ),
        start=1,
    ):
        section_output, section_debug = (
            _fit_section(
                gray,
                section,
                template,
            )
        )

        if section_output is None:
            if legacy_records is None:
                try:
                    (
                        legacy_records,
                        legacy_debug,
                    ) = (
                        _legacy_scan_jee_mcq_sections_robust(
                            corrected_image,
                            template,
                        )
                    )
                except Exception:
                    legacy_records = {}
                    legacy_debug = {}

            start_question = int(
                section["start_question"]
            )
            total_questions = int(
                section["total_questions"]
            )

            for row in range(total_questions):
                question = (
                    start_question + row
                )
                record = legacy_records.get(
                    question,
                    legacy_records.get(
                        str(question),
                    ),
                )

                if isinstance(record, dict):
                    output[question] = dict(record)

            section_debug = {
                **section_debug,
                "fallback":
                    "legacy_jee_grid_reader",
            }
        else:
            output.update(
                section_output
            )

        debug["sections"][
            str(section_index)
        ] = section_debug

    if legacy_debug is not None:
        debug[
            "legacy_fallback_debug"
        ] = legacy_debug

    return output, debug
