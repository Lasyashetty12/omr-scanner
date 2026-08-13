from __future__ import annotations

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
    Balanced mobile-photo decision engine.

    Key rule:
      - row-relative evidence may rescue ONE answer
      - MULTIPLE is never decided from relative evidence alone
      - every multiple bubble must independently be strongly filled

    This avoids the false red MULTIPLE explosion caused by the previous
    row-relative version.
    """

    darkness_threshold = float(
        sheet_thresholds[
            "filled_darkness_threshold"
        ]
    )

    core_ratio_threshold = float(
        sheet_thresholds[
            "filled_core_ratio_threshold"
        ]
    )

    ranked = sorted(
        option_data.items(),
        key=lambda item:
            float(
                item[1][
                    "metrics"
                ][
                    "center_darkness"
                ]
            ),
        reverse=True,
    )

    best_option, best_info = ranked[0]
    second_option, second_info = ranked[1]

    best_darkness = float(
        best_info[
            "metrics"
        ][
            "center_darkness"
        ]
    )

    second_darkness = float(
        second_info[
            "metrics"
        ][
            "center_darkness"
        ]
    )

    best_core_ratio = float(
        best_info[
            "metrics"
        ][
            "core_dark_ratio"
        ]
    )

    second_core_ratio = float(
        second_info[
            "metrics"
        ][
            "core_dark_ratio"
        ]
    )

    best_ml = float(
        best_info.get(
            "ml_filled_probability",
            0.0,
        )
    )

    second_ml = float(
        second_info.get(
            "ml_filled_probability",
            0.0,
        )
    )

    darkness_values = [
        float(
            info[
                "metrics"
            ][
                "center_darkness"
            ]
        )
        for _, info
        in ranked
    ]

    core_values = [
        float(
            info[
                "metrics"
            ][
                "core_dark_ratio"
            ]
        )
        for _, info
        in ranked
    ]

    question_blank_baseline = float(
        np.median(
            darkness_values[-2:]
        )
    )

    question_core_baseline = float(
        np.median(
            core_values[-2:]
        )
    )

    best_delta = (
        best_darkness
        -
        question_blank_baseline
    )

    second_delta = (
        second_darkness
        -
        question_blank_baseline
    )

    top_gap = (
        best_darkness
        -
        second_darkness
    )

    best_core_delta = (
        best_core_ratio
        -
        question_core_baseline
    )

    # --------------------------------------------------------
    # 1) STRICT MULTIPLE
    # --------------------------------------------------------
    # Both bubbles must independently look strongly filled.
    # Relative separation is NOT enough to create MULTIPLE.

    strong_multiple_options = []

    for option, info in ranked:
        metrics = info[
            "metrics"
        ]

        darkness = float(
            metrics[
                "center_darkness"
            ]
        )

        core_ratio = float(
            metrics[
                "core_dark_ratio"
            ]
        )

        ml_filled = float(
            info.get(
                "ml_filled_probability",
                0.0,
            )
        )

        delta = (
            darkness
            -
            question_blank_baseline
        )

        strong_absolute = (
            darkness
            >=
            max(
                48.0,
                darkness_threshold
                *
                0.86,
            )
            and
            core_ratio
            >=
            max(
                MULTIPLE_MIN_CORE_DARK_RATIO,
                core_ratio_threshold
                *
                0.82,
            )
            and
            delta
            >=
            MULTIPLE_MIN_DELTA
        )

        strong_ml_supported = (
            darkness
            >=
            max(
                44.0,
                darkness_threshold
                *
                0.78,
            )
            and
            core_ratio
            >=
            0.17
            and
            delta
            >=
            16.0
            and
            ml_filled
            >=
            0.82
        )

        if (
            strong_absolute
            or strong_ml_supported
        ):
            strong_multiple_options.append(
                option
            )

    if len(
        strong_multiple_options
    ) >= 2:
        return {
            "answer":
                "MULTIPLE",

            "status":
                "multiple",

            "multiple_options":
                strong_multiple_options,

            "best_option":
                best_option,

            "best_darkness":
                round(
                    best_darkness,
                    3,
                ),

            "second_darkness":
                round(
                    second_darkness,
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
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
    # 2) CLEAR SINGLE — row-relative rescue
    # --------------------------------------------------------
    # This is where row-relative logic helps mobile photos:
    # it may rescue the best option, but never create a second mark.

    clear_row_winner = (
        top_gap
        >=
        11.0
        and
        best_delta
        >=
        14.0
    )

    enough_core = (
        best_core_ratio
        >=
        max(
            0.15,
            core_ratio_threshold
            *
            0.65,
        )
        and
        best_core_delta
        >=
        0.035
    )

    enough_darkness = (
        best_darkness
        >=
        max(
            40.0,
            darkness_threshold
            *
            0.65,
        )
    )

    ml_support = (
        best_ml
        >=
        0.62
    )

    if (
        clear_row_winner
        and
        (
            enough_core
            or
            enough_darkness
            or
            ml_support
        )
    ):
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
                    best_darkness,
                    3,
                ),

            "second_darkness":
                round(
                    second_darkness,
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
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

            "row_relative_rescue":
                True,
        }

    # --------------------------------------------------------
    # 3) STRONG ABSOLUTE SINGLE
    # --------------------------------------------------------

    strong_absolute_single = (
        best_darkness
        >=
        max(
            46.0,
            darkness_threshold
            *
            0.82,
        )
        and
        best_core_ratio
        >=
        max(
            0.16,
            core_ratio_threshold
            *
            0.76,
        )
        and
        top_gap
        >=
        7.0
    )

    if strong_absolute_single:
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
                    best_darkness,
                    3,
                ),

            "second_darkness":
                round(
                    second_darkness,
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
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
    # 4) TRUE BLANK
    # --------------------------------------------------------

    blank_like = (
        best_delta
        <
        9.0
        and
        top_gap
        <
        8.0
        and
        best_core_ratio
        <
        max(
            0.15,
            core_ratio_threshold
            *
            0.72,
        )
        and
        best_darkness
        <
        max(
            40.0,
            darkness_threshold
            *
            0.74,
        )
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
                    best_darkness,
                    3,
                ),

            "second_darkness":
                round(
                    second_darkness,
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
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
    # 5) ML-assisted borderline SINGLE
    # --------------------------------------------------------

    if (
        best_ml
        >=
        0.70
        and
        best_delta
        >=
        10.0
        and
        top_gap
        >=
        8.0
        and
        best_core_ratio
        >=
        0.14
    ):
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
                    best_darkness,
                    3,
                ),

            "second_darkness":
                round(
                    second_darkness,
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
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

            "ml_relative_rescue":
                True,
        }

    # --------------------------------------------------------
    # 6) Borderline: ambiguous rather than false multiple/blank.
    # --------------------------------------------------------

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
                best_darkness,
                3,
            ),

        "second_darkness":
            round(
                second_darkness,
                3,
            ),

        "top_gap":
            round(
                top_gap,
                3,
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

    for question, option_data in (
        question_data.items()
    ):

        decision = _decide_question(
            option_data,
            sheet_thresholds,
        )

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

    return (
        answers,
        debug,
    )
