from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from ml_omr.inference import classify_batch


DEFAULT_CROP_RADIUS = 16

# ------------------------------------------------------------
# Tuned decision thresholds
# ------------------------------------------------------------

# Sheet-level adaptive threshold is still learned from blank bubbles.
MIN_FILLED_DARKNESS = 42.0
MIN_CORE_DARK_RATIO = 0.15

# Relative rescue: lightly filled bubbles can still be accepted when
# they are clearly darker than the other three options.
RELATIVE_RESCUE_MIN_GAP = 12.0
RELATIVE_RESCUE_ML = 0.65

# A true blank should have BOTH weak absolute evidence AND weak relative
# separation from the second-darkest bubble.
BLANK_ABSOLUTE_MARGIN = 0.84
BLANK_MAX_TOP_GAP = 9.0

# Multiple validation
MULTIPLE_MIN_DELTA = 18.0
MULTIPLE_MIN_CORE_DARK_RATIO = 0.19


def crop_bubble(
    gray,
    x,
    y,
    radius=DEFAULT_CROP_RADIUS,
):
    h, w = gray.shape[:2]

    x1 = max(0, int(round(x - radius)))
    y1 = max(0, int(round(y - radius)))
    x2 = min(w, int(round(x + radius + 1)))
    y2 = min(h, int(round(y + radius + 1)))

    return gray[y1:y2, x1:x2]



def refine_bubble_center(
    gray,
    x,
    y,
    search_radius_x=3,
    search_radius_y=1,
    ring_radius=10,
):
    """
    Tiny final refinement after column calibration.

    Horizontal movement is allowed up to +/-3 px.
    Vertical movement is limited to +/-1 px so the detector cannot
    drift downward into the next row or nearby printed lines.
    """

    h, w = gray.shape[:2]

    base_x = int(round(x))
    base_y = int(round(y))

    best_x = base_x
    best_y = base_y
    best_score = -1e9

    for dy in range(
        -search_radius_y,
        search_radius_y + 1,
    ):
        for dx in range(
            -search_radius_x,
            search_radius_x + 1,
        ):
            cx = base_x + dx
            cy = base_y + dy

            x1 = max(0, cx - ring_radius)
            y1 = max(0, cy - ring_radius)
            x2 = min(w, cx + ring_radius + 1)
            y2 = min(h, cy + ring_radius + 1)

            patch = gray[
                y1:y2,
                x1:x2,
            ]

            if (
                patch.shape[0] < 19
                or patch.shape[1] < 19
            ):
                continue

            ph, pw = patch.shape[:2]

            yy, xx = np.ogrid[
                :ph,
                :pw,
            ]

            pcx = (pw - 1) / 2.0
            pcy = (ph - 1) / 2.0

            rr = np.sqrt(
                (xx - pcx) ** 2
                +
                (yy - pcy) ** 2
            )

            ring_mask = (
                (rr >= ring_radius * 0.55)
                &
                (rr <= ring_radius * 0.88)
            )

            outside_mask = (
                (rr >= ring_radius * 0.92)
                &
                (rr <= ring_radius * 1.00)
            )

            if not np.any(
                ring_mask
            ):
                continue

            ring_mean = float(
                np.mean(
                    patch[
                        ring_mask
                    ]
                )
            )

            outside_mean = (
                float(
                    np.mean(
                        patch[
                            outside_mask
                        ]
                    )
                )
                if np.any(
                    outside_mask
                )
                else 230.0
            )

            score = (
                outside_mean
                -
                ring_mean
            )

            # Prefer staying near the calibrated coordinate.
            score -= abs(dx) * 0.35
            score -= abs(dy) * 1.50

            if score > best_score:
                best_score = score
                best_x = cx
                best_y = cy

    return (
        best_x,
        best_y,
    )



def _circle_mask(size, radius):
    center = (size - 1) / 2.0
    yy, xx = np.ogrid[:size, :size]

    return (
        (xx - center) ** 2
        +
        (yy - center) ** 2
        <= radius ** 2
    )


def _bubble_metrics(crop):
    if crop is None or crop.size == 0:
        return {
            "core_mean": 255.0,
            "paper_mean": 255.0,
            "center_darkness": 0.0,
            "core_dark_ratio": 0.0,
            "disk_dark_ratio": 0.0,
        }

    if crop.ndim == 3:
        crop = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

    crop = crop.astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=1.6,
        tileGridSize=(4, 4),
    )

    normalized = clahe.apply(crop)

    h, w = normalized.shape[:2]
    size = min(h, w)

    if size < 9:
        mean_value = float(np.mean(normalized))
        return {
            "core_mean": mean_value,
            "paper_mean": mean_value,
            "center_darkness": 0.0,
            "core_dark_ratio": 0.0,
            "disk_dark_ratio": 0.0,
        }

    y0 = (h - size) // 2
    x0 = (w - size) // 2

    square = normalized[
        y0:y0 + size,
        x0:x0 + size,
    ]

    core_radius = max(
        3.0,
        size * 0.17,
    )

    disk_radius = max(
        5.0,
        size * 0.30,
    )

    background_inner = size * 0.38
    background_outer = size * 0.49

    core_mask = _circle_mask(
        size,
        core_radius,
    )

    disk_mask = _circle_mask(
        size,
        disk_radius,
    )

    outer_mask = _circle_mask(
        size,
        background_outer,
    )

    inner_bg_mask = _circle_mask(
        size,
        background_inner,
    )

    background_mask = np.logical_and(
        outer_mask,
        np.logical_not(inner_bg_mask),
    )

    core_pixels = square[core_mask]
    disk_pixels = square[disk_mask]
    background_pixels = square[
        background_mask
    ]

    if core_pixels.size == 0:
        core_pixels = square.reshape(-1)

    if disk_pixels.size == 0:
        disk_pixels = square.reshape(-1)

    if background_pixels.size == 0:
        background_pixels = square.reshape(-1)

    paper_mean = float(
        np.percentile(
            background_pixels,
            70,
        )
    )

    core_mean = float(
        np.mean(
            core_pixels
        )
    )

    center_darkness = max(
        0.0,
        paper_mean - core_mean,
    )

    dark_threshold = int(
        np.clip(
            paper_mean - 42.0,
            80,
            165,
        )
    )

    core_dark_ratio = float(
        np.mean(
            core_pixels < dark_threshold
        )
    )

    disk_dark_ratio = float(
        np.mean(
            disk_pixels < dark_threshold
        )
    )

    return {
        "core_mean":
            round(
                core_mean,
                2,
            ),

        "paper_mean":
            round(
                paper_mean,
                2,
            ),

        "center_darkness":
            round(
                center_darkness,
                2,
            ),

        "core_dark_ratio":
            round(
                core_dark_ratio,
                4,
            ),

        "disk_dark_ratio":
            round(
                disk_dark_ratio,
                4,
            ),
    }


def _ml_probability(
    prediction,
    label,
):
    probabilities = prediction.get(
        "probabilities",
        {},
    )

    if (
        isinstance(probabilities, dict)
        and label in probabilities
    ):
        return float(
            probabilities[label]
        )

    predicted_label = str(
        prediction.get(
            "label",
            ""
        )
    ).lower()

    confidence = float(
        prediction.get(
            "confidence",
            0.0,
        )
    )

    if predicted_label == label:
        return confidence

    return 0.0


def _median_absolute_deviation(values):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if values.size == 0:
        return 0.0

    median = float(
        np.median(values)
    )

    return float(
        np.median(
            np.abs(
                values - median
            )
        )
    )


def _estimate_blank_distribution(
    all_question_data,
):
    blank_darkness_samples = []
    blank_core_ratio_samples = []

    for option_data in (
        all_question_data.values()
    ):
        ranked = sorted(
            option_data.values(),
            key=lambda item:
                float(
                    item[
                        "metrics"
                    ][
                        "center_darkness"
                    ]
                ),
        )

        for item in ranked[:2]:
            blank_darkness_samples.append(
                float(
                    item[
                        "metrics"
                    ][
                        "center_darkness"
                    ]
                )
            )

            blank_core_ratio_samples.append(
                float(
                    item[
                        "metrics"
                    ][
                        "core_dark_ratio"
                    ]
                )
            )

    blank_median = float(
        np.median(
            blank_darkness_samples
        )
    )

    blank_mad = (
        _median_absolute_deviation(
            blank_darkness_samples
        )
    )

    blank_ratio_median = float(
        np.median(
            blank_core_ratio_samples
        )
    )

    filled_darkness_threshold = max(
        MIN_FILLED_DARKNESS,
        blank_median
        +
        max(
            20.0,
            5.0 * blank_mad,
        ),
    )

    filled_core_ratio_threshold = max(
        MIN_CORE_DARK_RATIO,
        blank_ratio_median
        +
        0.10,
    )

    filled_darkness_threshold = float(
        np.clip(
            filled_darkness_threshold,
            46.0,
            110.0,
        )
    )

    filled_core_ratio_threshold = float(
        np.clip(
            filled_core_ratio_threshold,
            0.18,
            0.56,
        )
    )

    return {
        "blank_darkness_median":
            round(
                blank_median,
                3,
            ),

        "blank_darkness_mad":
            round(
                blank_mad,
                3,
            ),

        "blank_core_ratio_median":
            round(
                blank_ratio_median,
                4,
            ),

        "filled_darkness_threshold":
            round(
                filled_darkness_threshold,
                3,
            ),

        "filled_core_ratio_threshold":
            round(
                filled_core_ratio_threshold,
                4,
            ),
    }


def _decide_question(
    option_data,
    sheet_thresholds,
):
    """
    Disk-first mobile-photo decision engine.

    Why:
    - Printed empty bubbles can have high core_dark_ratio because the ring
      and digits are dark.
    - A genuinely filled bubble usually darkens a much larger fraction of
      the bubble disk.
    - Therefore disk_dark_ratio is the primary evidence for SINGLE/MULTIPLE.
    - Row-relative darkness is used only as a conservative faint-mark rescue.
    - ML is supporting evidence only.
    """

    del sheet_thresholds

    ranked = sorted(
        option_data.items(),
        key=lambda item: (
            float(
                item[1]["metrics"]["disk_dark_ratio"]
            ),
            float(
                item[1]["metrics"]["center_darkness"]
            ),
        ),
        reverse=True,
    )

    best_option, best_info = ranked[0]
    second_option, second_info = ranked[1]

    def metrics_for(info):
        metrics = info["metrics"]
        return {
            "darkness": float(
                metrics["center_darkness"]
            ),
            "core": float(
                metrics["core_dark_ratio"]
            ),
            "disk": float(
                metrics["disk_dark_ratio"]
            ),
            "ml": float(
                info.get(
                    "ml_filled_probability",
                    0.0,
                )
            ),
        }

    best = metrics_for(best_info)
    second = metrics_for(second_info)

    darkness_values = sorted(
        [
            float(
                info["metrics"]["center_darkness"]
            )
            for _, info in option_data.items()
        ]
    )

    disk_values = sorted(
        [
            float(
                info["metrics"]["disk_dark_ratio"]
            )
            for _, info in option_data.items()
        ]
    )

    question_blank_baseline = float(
        np.median(
            darkness_values[:2]
        )
    )

    disk_blank_baseline = float(
        np.median(
            disk_values[:2]
        )
    )

    best_delta = (
        best["darkness"]
        -
        question_blank_baseline
    )

    top_gap = (
        best["darkness"]
        -
        second["darkness"]
    )

    disk_gap = (
        best["disk"]
        -
        second["disk"]
    )

    # --------------------------------------------------------
    # Strong filled bubble
    # --------------------------------------------------------
    # The report shows genuine full marks are typically near disk=1.0,
    # while many false "second fills" are around 0.40-0.65.
    #
    # Keep multiple very strict.
    def is_strong_fill(info):
        m = metrics_for(info)

        return (
            m["disk"] >= 0.78
            and
            m["core"] >= 0.72
            and
            m["darkness"] >= 82.0
        )

    strong_options = [
        option
        for option, info
        in option_data.items()
        if is_strong_fill(
            info
        )
    ]

    # TRUE MULTIPLE:
    # at least two independently full-disk dark bubbles.
    if len(
        strong_options
    ) >= 2:
        return {
            "answer":
                "MULTIPLE",

            "status":
                "multiple",

            "multiple_options":
                strong_options,

            "best_option":
                best_option,

            "best_darkness":
                round(
                    best["darkness"],
                    3,
                ),

            "second_darkness":
                round(
                    second["darkness"],
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
                ),

            "best_disk_ratio":
                round(
                    best["disk"],
                    4,
                ),

            "second_disk_ratio":
                round(
                    second["disk"],
                    4,
                ),

            "disk_gap":
                round(
                    disk_gap,
                    4,
                ),

            "question_blank_baseline":
                round(
                    question_blank_baseline,
                    3,
                ),

            "best_delta":
                round(
                    best_delta,
                    3,
                ),
        }

    # One independently strong fill -> SINGLE immediately.
    if len(
        strong_options
    ) == 1:
        winner = strong_options[0]

        return {
            "answer":
                winner,

            "status":
                "answered",

            "multiple_options":
                [],

            "best_option":
                winner,

            "best_darkness":
                round(
                    best["darkness"],
                    3,
                ),

            "second_darkness":
                round(
                    second["darkness"],
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
                ),

            "best_disk_ratio":
                round(
                    best["disk"],
                    4,
                ),

            "second_disk_ratio":
                round(
                    second["disk"],
                    4,
                ),

            "disk_gap":
                round(
                    disk_gap,
                    4,
                ),

            "question_blank_baseline":
                round(
                    question_blank_baseline,
                    3,
                ),

            "best_delta":
                round(
                    best_delta,
                    3,
                ),

            "disk_first":
                True,
        }

    # --------------------------------------------------------
    # Medium fill — still require substantial disk coverage.
    # --------------------------------------------------------
    medium_fill = (
        best["disk"] >= 0.68
        and
        best["core"] >= 0.62
        and
        best["darkness"] >= 68.0
        and
        (
            disk_gap >= 0.10
            or
            top_gap >= 12.0
        )
    )

    if medium_fill:
        return {
            "answer":
                best_option,

            "status":
                "answered",

            "multiple_options":
                [],

            "best_option":
                best_option,

            "best_darkness":
                round(
                    best["darkness"],
                    3,
                ),

            "second_darkness":
                round(
                    second["darkness"],
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
                ),

            "best_disk_ratio":
                round(
                    best["disk"],
                    4,
                ),

            "second_disk_ratio":
                round(
                    second["disk"],
                    4,
                ),

            "disk_gap":
                round(
                    disk_gap,
                    4,
                ),

            "question_blank_baseline":
                round(
                    question_blank_baseline,
                    3,
                ),

            "best_delta":
                round(
                    best_delta,
                    3,
                ),

            "medium_disk_rescue":
                True,
        }

    # --------------------------------------------------------
    # Conservative faint-mark rescue.
    # --------------------------------------------------------
    # Only one answer can be rescued this way. This rule can NEVER create
    # MULTIPLE.
    faint_relative = (
        best["disk"] >= 0.54
        and
        best["core"] >= 0.55
        and
        best["darkness"] >= 55.0
        and
        best_delta >= 15.0
        and
        top_gap >= 10.0
        and
        (
            best["ml"] >= 0.65
            or
            disk_gap >= 0.08
        )
    )

    if faint_relative:
        return {
            "answer":
                best_option,

            "status":
                "answered",

            "multiple_options":
                [],

            "best_option":
                best_option,

            "best_darkness":
                round(
                    best["darkness"],
                    3,
                ),

            "second_darkness":
                round(
                    second["darkness"],
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
                ),

            "best_disk_ratio":
                round(
                    best["disk"],
                    4,
                ),

            "second_disk_ratio":
                round(
                    second["disk"],
                    4,
                ),

            "disk_gap":
                round(
                    disk_gap,
                    4,
                ),

            "question_blank_baseline":
                round(
                    question_blank_baseline,
                    3,
                ),

            "best_delta":
                round(
                    best_delta,
                    3,
                ),

            "faint_relative_rescue":
                True,
        }

    # --------------------------------------------------------
    # True blank
    # --------------------------------------------------------
    # Blank rows have no substantial full-disk darkening.
    blank_like = (
        best["disk"] < 0.58
        and
        best["darkness"] < 62.0
        and
        best_delta < 14.0
    )

    if blank_like:
        return {
            "answer":
                None,

            "status":
                "blank",

            "multiple_options":
                [],

            "best_option":
                best_option,

            "best_darkness":
                round(
                    best["darkness"],
                    3,
                ),

            "second_darkness":
                round(
                    second["darkness"],
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
                ),

            "best_disk_ratio":
                round(
                    best["disk"],
                    4,
                ),

            "second_disk_ratio":
                round(
                    second["disk"],
                    4,
                ),

            "disk_gap":
                round(
                    disk_gap,
                    4,
                ),

            "question_blank_baseline":
                round(
                    question_blank_baseline,
                    3,
                ),

            "best_delta":
                round(
                    best_delta,
                    3,
                ),
        }

    # --------------------------------------------------------
    # Ambiguous-only surgical rescue
    # --------------------------------------------------------
    # This runs ONLY after all existing 419-baseline rules have failed.
    # It does not change already-answered / blank / multiple rows.
    #
    # Rescue only when the SAME option is the winner by BOTH:
    #   1) center darkness
    #   2) disk dark ratio
    #
    # and that option has enough separation from the runners-up.
    #
    # This targets the unstable ambiguous rows seen in the 419 baseline
    # without changing the normal decision path.

    by_darkness = sorted(
        option_data.items(),
        key=lambda item: float(
            item[1]["metrics"]["center_darkness"]
        ),
        reverse=True,
    )

    by_disk = sorted(
        option_data.items(),
        key=lambda item: float(
            item[1]["metrics"]["disk_dark_ratio"]
        ),
        reverse=True,
    )

    darkness_winner = by_darkness[0][0]
    disk_winner = by_disk[0][0]

    if darkness_winner == disk_winner:
        winner = darkness_winner

        winner_info = option_data[winner]
        winner_metrics = winner_info["metrics"]

        winner_darkness = float(
            winner_metrics["center_darkness"]
        )
        winner_disk = float(
            winner_metrics["disk_dark_ratio"]
        )
        winner_core = float(
            winner_metrics["core_dark_ratio"]
        )
        winner_ml = float(
            winner_info.get(
                "ml_filled_probability",
                0.0,
            )
        )

        second_darkness_same_metric = float(
            by_darkness[1][1]["metrics"]["center_darkness"]
        )

        second_disk_same_metric = float(
            by_disk[1][1]["metrics"]["disk_dark_ratio"]
        )

        darkness_gap = (
            winner_darkness
            -
            second_darkness_same_metric
        )

        disk_gap_same_metric = (
            winner_disk
            -
            second_disk_same_metric
        )

        # Conservative rescue:
        # - same winner on two independent image features
        # - enough absolute evidence
        # - enough separation on at least one feature
        #
        # ML is supporting evidence only; it cannot rescue on its own.
        ambiguous_rescue = (
            winner_darkness >= 62.0
            and
            winner_disk >= 0.60
            and
            winner_core >= 0.62
            and
            (
                darkness_gap >= 6.0
                or
                disk_gap_same_metric >= 0.045
            )
            and
            (
                winner_ml >= 0.80
                or
                darkness_gap >= 10.0
                or
                disk_gap_same_metric >= 0.075
            )
        )

        if ambiguous_rescue:
            return {
                "answer":
                    winner,

                "status":
                    "answered",

                "multiple_options":
                    [],

                "best_option":
                    winner,

                "best_darkness":
                    round(
                        winner_darkness,
                        3,
                    ),

                "second_darkness":
                    round(
                        second_darkness_same_metric,
                        3,
                    ),

                "top_gap":
                    round(
                        darkness_gap,
                        3,
                    ),

                "best_disk_ratio":
                    round(
                        winner_disk,
                        4,
                    ),

                "second_disk_ratio":
                    round(
                        second_disk_same_metric,
                        4,
                    ),

                "disk_gap":
                    round(
                        disk_gap_same_metric,
                        4,
                    ),

                "question_blank_baseline":
                    round(
                        question_blank_baseline,
                        3,
                    ),

                "best_delta":
                    round(
                        (
                            winner_darkness
                            -
                            question_blank_baseline
                        ),
                        3,
                    ),

                "ambiguous_rescue":
                    True,
            }

    # Borderline rows remain uncertain instead of inventing a result.
    return {
        "answer":
            None,

        "status":
            "ambiguous",

        "multiple_options":
            [],

        "best_option":
            best_option,

        "best_darkness":
            round(
                best["darkness"],
                3,
            ),

        "second_darkness":
            round(
                second["darkness"],
                3,
            ),

        "top_gap":
            round(
                top_gap,
                3,
            ),

        "best_disk_ratio":
            round(
                best["disk"],
                4,
            ),

        "second_disk_ratio":
            round(
                second["disk"],
                4,
            ),

        "disk_gap":
            round(
                disk_gap,
                4,
            ),

        "question_blank_baseline":
            round(
                question_blank_baseline,
                3,
            ),

        "best_delta":
            round(
                best_delta,
                3,
            ),
    }



def export_recognition_report(
    question_data,
    decisions,
    sheet_thresholds,
    csv_path="recognition_report.csv",
    json_path="recognition_report.json",
):
    """
    Export one row per question-option plus the final decision.

    CSV columns include:
      question, option,
      center_darkness, core_dark_ratio, disk_dark_ratio,
      ml_filled_probability, ml_blank_probability,
      ml_ambiguous_probability,
      crop_center_x, crop_center_y,
      final_answer, final_status,
      best_option, best_darkness, second_darkness,
      top_gap, question_blank_baseline, best_delta,
      sheet filled thresholds.

    JSON keeps the full nested debug structure.
    """

    csv_file = Path(
        csv_path
    )

    json_file = Path(
        json_path
    )

    rows = []

    for question, option_data in question_data.items():
        decision = decisions[
            question
        ]

        for option, info in option_data.items():
            metrics = info.get(
                "metrics",
                {},
            )

            crop_center = info.get(
                "crop_center",
                [
                    None,
                    None,
                ],
            )

            rows.append(
                {
                    "question":
                        int(
                            question
                        ),

                    "option":
                        str(
                            option
                        ),

                    "center_darkness":
                        metrics.get(
                            "center_darkness"
                        ),

                    "core_dark_ratio":
                        metrics.get(
                            "core_dark_ratio"
                        ),

                    "disk_dark_ratio":
                        metrics.get(
                            "disk_dark_ratio"
                        ),

                    "core_mean":
                        metrics.get(
                            "core_mean"
                        ),

                    "paper_mean":
                        metrics.get(
                            "paper_mean"
                        ),

                    "ml_filled_probability":
                        info.get(
                            "ml_filled_probability"
                        ),

                    "ml_blank_probability":
                        info.get(
                            "ml_blank_probability"
                        ),

                    "ml_ambiguous_probability":
                        info.get(
                            "ml_ambiguous_probability"
                        ),

                    "crop_center_x":
                        crop_center[
                            0
                        ]
                        if len(
                            crop_center
                        )
                        > 0
                        else None,

                    "crop_center_y":
                        crop_center[
                            1
                        ]
                        if len(
                            crop_center
                        )
                        > 1
                        else None,

                    "final_answer":
                        decision.get(
                            "answer"
                        ),

                    "final_status":
                        decision.get(
                            "status"
                        ),

                    "best_option":
                        decision.get(
                            "best_option"
                        ),

                    "best_darkness":
                        decision.get(
                            "best_darkness"
                        ),

                    "second_darkness":
                        decision.get(
                            "second_darkness"
                        ),

                    "top_gap":
                        decision.get(
                            "top_gap"
                        ),

                    "question_blank_baseline":
                        decision.get(
                            "question_blank_baseline"
                        ),

                    "best_delta":
                        decision.get(
                            "best_delta"
                        ),

                    "sheet_blank_darkness_median":
                        sheet_thresholds.get(
                            "blank_darkness_median"
                        ),

                    "sheet_blank_darkness_mad":
                        sheet_thresholds.get(
                            "blank_darkness_mad"
                        ),

                    "sheet_filled_darkness_threshold":
                        sheet_thresholds.get(
                            "filled_darkness_threshold"
                        ),

                    "sheet_filled_core_ratio_threshold":
                        sheet_thresholds.get(
                            "filled_core_ratio_threshold"
                        ),
                }
            )

    fieldnames = [
        "question",
        "option",
        "center_darkness",
        "core_dark_ratio",
        "disk_dark_ratio",
        "core_mean",
        "paper_mean",
        "ml_filled_probability",
        "ml_blank_probability",
        "ml_ambiguous_probability",
        "crop_center_x",
        "crop_center_y",
        "final_answer",
        "final_status",
        "best_option",
        "best_darkness",
        "second_darkness",
        "top_gap",
        "question_blank_baseline",
        "best_delta",
        "sheet_blank_darkness_median",
        "sheet_blank_darkness_mad",
        "sheet_filled_darkness_threshold",
        "sheet_filled_core_ratio_threshold",
    ]

    with csv_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    json_payload = {
        "sheet_thresholds":
            sheet_thresholds,

        "questions":
            {
                str(
                    question
                ):
                    {
                        "decision":
                            decisions[
                                question
                            ],

                        "options":
                            option_data,
                    }
                for question, option_data
                in question_data.items()
            },
    }

    with json_file.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "csv":
            str(
                csv_file
            ),

        "json":
            str(
                json_file
            ),

        "row_count":
            len(
                rows
            ),
    }


def scan_answers_ml(
    gray,
    coordinates,
    crop_radius=DEFAULT_CROP_RADIUS,
    filled_confidence=0.70,
    ambiguous_confidence=0.60,
):
    """
    Final tuned adaptive hybrid reader.

    Main changes:
      - BLANK needs weak absolute AND weak relative evidence
      - faint but clearly separated fills can be rescued
      - MULTIPLE still requires independently strong bubbles
      - ML remains supporting evidence, not the scoring engine
    """

    del filled_confidence
    del ambiguous_confidence

    if gray.ndim == 3:
        gray = cv2.cvtColor(
            gray,
            cv2.COLOR_BGR2GRAY,
        )

    batch_crops = []
    batch_map = []

    question_data = {}

    for question, option_map in (
        coordinates.items()
    ):

        question_data[
            question
        ] = {}

        for option, (
            x,
            y,
        ) in option_map.items():

            # The grid detector already fitted the actual printed bubble
            # lattice. Do NOT move the center again here, otherwise the
            # final local search can drift away from the fitted circle.
            refined_x = int(round(x))
            refined_y = int(round(y))

            crop = crop_bubble(
                gray,
                refined_x,
                refined_y,
                crop_radius,
            )

            metrics = _bubble_metrics(
                crop
            )

            question_data[
                question
            ][
                option
            ] = {
                "metrics":
                    metrics,

                "crop_center": [
                    int(refined_x),
                    int(refined_y),
                ],

                "calibrated_center": [
                    int(round(x)),
                    int(round(y)),
                ],
            }

            batch_crops.append(
                crop
            )

            batch_map.append(
                (
                    question,
                    option,
                )
            )

    predictions = classify_batch(
        batch_crops
    )

    for (
        question,
        option,
    ), prediction in zip(
        batch_map,
        predictions,
    ):

        question_data[
            question
        ][
            option
        ][
            "ml"
        ] = prediction

        question_data[
            question
        ][
            option
        ][
            "ml_filled_probability"
        ] = round(
            _ml_probability(
                prediction,
                "filled",
            ),
            4,
        )

        question_data[
            question
        ][
            option
        ][
            "ml_blank_probability"
        ] = round(
            _ml_probability(
                prediction,
                "blank",
            ),
            4,
        )

        question_data[
            question
        ][
            option
        ][
            "ml_ambiguous_probability"
        ] = round(
            _ml_probability(
                prediction,
                "ambiguous",
            ),
            4,
        )

    sheet_thresholds = (
        _estimate_blank_distribution(
            question_data
        )
    )

    answers = {}
    debug = {}
    decisions = {}

    for question, option_data in (
        question_data.items()
    ):

        decision = _decide_question(
            option_data,
            sheet_thresholds,
        )

        decisions[
            question
        ] = decision

        answers[
            question
        ] = decision[
            "answer"
        ]

        debug[
            question
        ] = {
            **decision,

            "sheet_thresholds":
                sheet_thresholds,

            "options":
                option_data,
        }

    try:
        report_info = (
            export_recognition_report(
                question_data=
                    question_data,

                decisions=
                    decisions,

                sheet_thresholds=
                    sheet_thresholds,

                csv_path=
                    "recognition_report.csv",

                json_path=
                    "recognition_report.json",
            )
        )

        debug[
            "_recognition_report"
        ] = report_info

    except Exception as report_error:
        debug[
            "_recognition_report"
        ] = {
            "error":
                str(
                    report_error
                ),
        }

    return (
        answers,
        debug,
    )
