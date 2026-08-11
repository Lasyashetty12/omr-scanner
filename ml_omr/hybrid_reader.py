from __future__ import annotations

import cv2
import numpy as np

from ml_omr.inference import classify_batch


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_CROP_RADIUS = 16

# Absolute safeguards. The final threshold is adapted per sheet.
MIN_FILLED_DARKNESS = 48.0
MIN_QUESTION_DELTA = 24.0
MIN_CORE_DARK_RATIO = 0.20

# A second marked bubble must also independently pass these checks
# before we call a question MULTIPLE.
MULTIPLE_MIN_DELTA = 20.0
MULTIPLE_MIN_CORE_DARK_RATIO = 0.18


# ============================================================
# CROP / MASK HELPERS
# ============================================================

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


def _circle_mask(size, radius):
    center = (size - 1) / 2.0

    yy, xx = np.ogrid[:size, :size]

    return (
        (xx - center) ** 2
        +
        (yy - center) ** 2
        <= radius ** 2
    )


# ============================================================
# BUBBLE METRICS
# ============================================================

def _bubble_metrics(crop):
    """
    Extract features that separate a real filled bubble from a printed
    empty outline.

    The CENTER of the bubble matters most. Empty printed outlines are
    dark mostly around the outer ring while their center remains bright.
    """

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

    # Mild normalization only.
    clahe = cv2.createCLAHE(
        clipLimit=1.6,
        tileGridSize=(4, 4),
    )

    normalized = clahe.apply(crop)

    h, w = normalized.shape[:2]
    size = min(h, w)

    if size < 9:
        return {
            "core_mean": float(np.mean(normalized)),
            "paper_mean": float(np.mean(normalized)),
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

    # Inner region deliberately avoids most of the printed ring.
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
        np.logical_not(
            inner_bg_mask
        ),
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

    # Main feature: how much darker the center is than its local paper.
    center_darkness = max(
        0.0,
        paper_mean - core_mean,
    )

    # Local threshold adapts to shadows / exposure.
    dark_threshold = int(
        np.clip(
            paper_mean - 42.0,
            80,
            165,
        )
    )

    core_dark_ratio = float(
        np.mean(
            core_pixels
            < dark_threshold
        )
    )

    disk_dark_ratio = float(
        np.mean(
            disk_pixels
            < dark_threshold
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


# ============================================================
# ML HELPERS
# ============================================================

def _ml_probability(
    prediction,
    label,
):
    probabilities = prediction.get(
        "probabilities",
        {},
    )

    if (
        isinstance(
            probabilities,
            dict,
        )
        and label in probabilities
    ):
        return float(
            probabilities[
                label
            ]
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


# ============================================================
# ADAPTIVE SHEET THRESHOLD
# ============================================================

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
    """
    Estimate what EMPTY bubbles look like on THIS scanned sheet.

    For every question, the two least-dark bubbles are almost always empty
    even when the question is answered or multiple-marked. Using them over
    180 questions gives a very stable blank baseline without needing a
    hard-coded brightness threshold.
    """

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

        # The two weakest options are the safest blank samples.
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

    blank_mad = _median_absolute_deviation(
        blank_darkness_samples
    )

    blank_ratio_median = float(
        np.median(
            blank_core_ratio_samples
        )
    )

    # Filled threshold is derived from actual empty bubbles on this image.
    #
    # The fixed floor protects against a very noisy/shadowy sheet.
    filled_darkness_threshold = max(
        MIN_FILLED_DARKNESS,
        blank_median
        +
        max(
            22.0,
            6.0 * blank_mad,
        ),
    )

    filled_core_ratio_threshold = max(
        MIN_CORE_DARK_RATIO,
        blank_ratio_median
        + 0.12,
    )

    # Keep within sensible ranges.
    filled_darkness_threshold = float(
        np.clip(
            filled_darkness_threshold,
            48.0,
            115.0,
        )
    )

    filled_core_ratio_threshold = float(
        np.clip(
            filled_core_ratio_threshold,
            0.20,
            0.60,
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


# ============================================================
# QUESTION DECISION
# ============================================================

def _decide_question(
    option_data,
    sheet_thresholds,
):
    """
    Decide one A/B/C/D question.

    Decision priority:
      1. absolute center-fill evidence
      2. difference from the other options in the SAME question
      3. ML used only as tie/support evidence
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

    darkness_values = [
        float(
            item[
                "metrics"
            ][
                "center_darkness"
            ]
        )
        for _, item
        in ranked
    ]

    # Robust within-question blank baseline:
    # use the median of the two least-dark bubbles.
    question_blank_baseline = float(
        np.median(
            darkness_values[
                -2:
            ]
        )
    )

    qualified = []

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

        delta = (
            darkness
            -
            question_blank_baseline
        )

        ml_filled = float(
            info[
                "ml_filled_probability"
            ]
        )

        # Main deterministic fill test.
        passes_classical = (
            darkness
            >= darkness_threshold
            and
            delta
            >= MIN_QUESTION_DELTA
            and
            core_ratio
            >= core_ratio_threshold
        )

        # Rescue a slightly weaker classical bubble only if ML is very sure.
        passes_ml_rescue = (
            darkness
            >= darkness_threshold * 0.82
            and
            delta
            >= MIN_QUESTION_DELTA * 0.82
            and
            core_ratio
            >= core_ratio_threshold * 0.82
            and
            ml_filled
            >= 0.92
        )

        is_filled = (
            passes_classical
            or passes_ml_rescue
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
            "passes_classical"
        ] = bool(
            passes_classical
        )

        info[
            "passes_ml_rescue"
        ] = bool(
            passes_ml_rescue
        )

        info[
            "is_filled"
        ] = bool(
            is_filled
        )

        if is_filled:
            qualified.append(
                option
            )

    # --------------------------------------------------------
    # BLANK
    # --------------------------------------------------------

    if len(qualified) == 0:
        return {
            "answer":
                None,

            "status":
                "blank",

            "multiple_options":
                [],

            "best_option":
                ranked[0][0],

            "question_blank_baseline":
                round(
                    question_blank_baseline,
                    3,
                ),
        }

    # --------------------------------------------------------
    # SINGLE
    # --------------------------------------------------------

    if len(qualified) == 1:
        return {
            "answer":
                qualified[0],

            "status":
                "answered",

            "multiple_options":
                [],

            "best_option":
                qualified[0],

            "question_blank_baseline":
                round(
                    question_blank_baseline,
                    3,
                ),
        }

    # --------------------------------------------------------
    # TRUE MULTIPLE
    # --------------------------------------------------------

    # Re-validate every additional marked bubble independently.
    # This suppresses accidental MULTIPLE caused by an empty outline.
    strong_multiple = []

    for option in qualified:

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
                ranked[0][0],

            "question_blank_baseline":
                round(
                    question_blank_baseline,
                    3,
                ),
        }

    # If only one bubble survives the stricter multiple check,
    # treat it as a normal single answer.
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

            "question_blank_baseline":
                round(
                    question_blank_baseline,
                    3,
                ),
        }

    # Conservative fallback.
    return {
        "answer":
            None,

        "status":
            "ambiguous",

        "multiple_options":
            [],

        "best_option":
            ranked[0][0],

        "question_blank_baseline":
            round(
                question_blank_baseline,
                3,
            ),
    }


# ============================================================
# PUBLIC API
# ============================================================

def scan_answers_ml(
    gray,
    coordinates,
    crop_radius=DEFAULT_CROP_RADIUS,
    filled_confidence=0.70,
    ambiguous_confidence=0.60,
):
    """
    Adaptive hybrid OMR reader.

    `filled_confidence` and `ambiguous_confidence` are kept in the public
    signature for compatibility with scanner.py.

    The reader adapts its blank/filled threshold to every scanned sheet.
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

    # --------------------------------------------------------
    # Extract classical metrics and prepare one ML batch
    # --------------------------------------------------------

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

            crop = crop_bubble(
                gray,
                x,
                y,
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
                    int(
                        round(x)
                    ),
                    int(
                        round(y)
                    ),
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

    # --------------------------------------------------------
    # ML inference
    # --------------------------------------------------------

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

        ml_filled = _ml_probability(
            prediction,
            "filled",
        )

        ml_blank = _ml_probability(
            prediction,
            "blank",
        )

        ml_ambiguous = _ml_probability(
            prediction,
            "ambiguous",
        )

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
            ml_filled,
            4,
        )

        question_data[
            question
        ][
            option
        ][
            "ml_blank_probability"
        ] = round(
            ml_blank,
            4,
        )

        question_data[
            question
        ][
            option
        ][
            "ml_ambiguous_probability"
        ] = round(
            ml_ambiguous,
            4,
        )

    # --------------------------------------------------------
    # Learn empty-bubble characteristics from this sheet
    # --------------------------------------------------------

    sheet_thresholds = (
        _estimate_blank_distribution(
            question_data
        )
    )

    answers = {}
    debug = {}

    # --------------------------------------------------------
    # Decide every question
    # --------------------------------------------------------

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
