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
RELATIVE_RESCUE_MIN_GAP = 14.0
RELATIVE_RESCUE_ML = 0.72

# A true blank should have BOTH weak absolute evidence AND weak relative
# separation from the second-darkest bubble.
BLANK_ABSOLUTE_MARGIN = 0.88
BLANK_MAX_TOP_GAP = 14.0

# Multiple validation
MULTIPLE_MIN_DELTA = 20.0
MULTIPLE_MIN_CORE_DARK_RATIO = 0.18


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

    top_gap = (
        best_darkness
        -
        second_darkness
    )

    best_core_ratio = float(
        best_info[
            "metrics"
        ][
            "core_dark_ratio"
        ]
    )

    best_ml = float(
        best_info[
            "ml_filled_probability"
        ]
    )

    # Robust within-question blank baseline:
    # median of the two least-dark options.
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

    question_blank_baseline = float(
        np.median(
            darkness_values[-2:]
        )
    )

    best_delta = (
        best_darkness
        -
        question_blank_baseline
    )

    # --------------------------------------------------------
    # TRUE BLANK
    # --------------------------------------------------------
    #
    # IMPORTANT CHANGE:
    # We now require BOTH weak absolute evidence AND a small top gap.
    #
    # This avoids throwing away lightly shaded real answers that are
    # clearly darker than the other three bubbles.

    weak_absolute = (
        best_darkness
        <
        darkness_threshold
        *
        BLANK_ABSOLUTE_MARGIN
    )

    weak_relative = (
        top_gap
        <
        BLANK_MAX_TOP_GAP
    )

    weak_core = (
        best_core_ratio
        <
        core_ratio_threshold
        *
        0.90
    )

    if (
        weak_absolute
        and
        weak_relative
        and
        weak_core
    ):
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
        }

    # --------------------------------------------------------
    # MARKED OPTIONS
    # --------------------------------------------------------

    filled_options = []

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
            info[
                "ml_filled_probability"
            ]
        )

        delta = (
            darkness
            -
            question_blank_baseline
        )

        absolute_pass = (
            darkness
            >=
            darkness_threshold
            and
            core_ratio
            >=
            core_ratio_threshold
        )

        # Relative rescue:
        # slightly faint bubble, but clearly darkest in the row and ML agrees.
        relative_rescue = (
            option == best_option
            and
            top_gap
            >=
            RELATIVE_RESCUE_MIN_GAP
            and
            delta
            >=
            RELATIVE_RESCUE_MIN_GAP
            and
            darkness
            >=
            darkness_threshold
            *
            0.72
            and
            core_ratio
            >=
            core_ratio_threshold
            *
            0.70
            and
            ml_filled
            >=
            RELATIVE_RESCUE_ML
        )

        # Very clear classical rescue even when ML is uncertain.
        strong_relative_rescue = (
            option == best_option
            and
            top_gap
            >=
            RELATIVE_RESCUE_MIN_GAP
            *
            1.35
            and
            delta
            >=
            RELATIVE_RESCUE_MIN_GAP
            *
            1.35
            and
            darkness
            >=
            darkness_threshold
            *
            0.80
            and
            core_ratio
            >=
            core_ratio_threshold
            *
            0.78
        )

        is_filled = (
            absolute_pass
            or relative_rescue
            or strong_relative_rescue
        )

        info[
            "question_blank_baseline"
        ] = round(
            question_blank_baseline,
            3,
        )

        info[
            "question_delta"
        ] = round(
            delta,
            3,
        )

        info[
            "absolute_pass"
        ] = bool(
            absolute_pass
        )

        info[
            "relative_rescue"
        ] = bool(
            relative_rescue
        )

        info[
            "strong_relative_rescue"
        ] = bool(
            strong_relative_rescue
        )

        info[
            "is_filled"
        ] = bool(
            is_filled
        )

        if is_filled:
            filled_options.append(
                option
            )

    # --------------------------------------------------------
    # NO FILLED OPTION AFTER RESCUE
    # --------------------------------------------------------

    if len(
        filled_options
    ) == 0:

        # ----------------------------------------------------
        # STRONG DARK-MARK RESCUE
        # ----------------------------------------------------
        # Mobile photos can shift local brightness enough that a visibly
        # black filled bubble narrowly misses the adaptive sheet threshold.
        # If the best option is still strongly dark, has a dark core, and
        # is clearly separated from the other options, accept it directly.
        strong_dark_mark = (
            best_darkness
            >=
            max(
                54.0,
                darkness_threshold
                *
                0.82,
            )
            and
            best_core_ratio
            >=
            max(
                0.18,
                core_ratio_threshold
                *
                0.78,
            )
            and
            top_gap
            >=
            11.0
            and
            best_delta
            >=
            14.0
        )

        if strong_dark_mark:
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

                "strong_dark_rescue":
                    True,
            }

        # If the top bubble is clearly separated, prefer UNCERTAIN rather
        # than falsely calling a real faint mark blank.
        if (
            top_gap
            >=
            RELATIVE_RESCUE_MIN_GAP
            and
            best_delta
            >=
            RELATIVE_RESCUE_MIN_GAP
        ):
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
            }

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
        }

    # --------------------------------------------------------
    # SINGLE
    # --------------------------------------------------------

    if len(
        filled_options
    ) == 1:
        return {
            "answer":
                filled_options[0],

            "status":
                "answered",

            "multiple_options":
                [],

            "best_option":
                filled_options[0],

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
        }

    # --------------------------------------------------------
    # MULTIPLE
    # --------------------------------------------------------

    strong_multiple = []

    for option in filled_options:

        info = option_data[
            option
        ]

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

        delta = (
            darkness
            -
            question_blank_baseline
        )

        if (
            darkness
            >= darkness_threshold
            and
            delta
            >= MULTIPLE_MIN_DELTA
            and
            core_ratio
            >= MULTIPLE_MIN_CORE_DARK_RATIO
        ):
            strong_multiple.append(
                option
            )

    if len(
        strong_multiple
    ) >= 2:
        return {
            "answer":
                "MULTIPLE",

            "status":
                "multiple",

            "multiple_options":
                strong_multiple,

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
        }

    # If only one option survives the strict multiple validation,
    # keep it as a normal single.
    if len(
        strong_multiple
    ) == 1:
        return {
            "answer":
                strong_multiple[0],

            "status":
                "answered",

            "multiple_options":
                [],

            "best_option":
                strong_multiple[0],

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
        }

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
