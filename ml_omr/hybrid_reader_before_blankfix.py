from __future__ import annotations

import cv2
import numpy as np

from ml_omr.inference import classify_batch


# ============================================================
# CONFIG
# ============================================================

DEFAULT_CROP_RADIUS = 16

# Per-question relative-decision thresholds.
# These are intentionally conservative because the corrected sheet
# is already geometrically aligned to the canonical template.
SINGLE_MIN_SCORE = 0.50
SINGLE_MIN_GAP = 0.12

MULTI_MIN_SCORE = 0.50
MULTI_MAX_GAP_BETWEEN_TOP_TWO = 0.11
MULTI_MIN_GAP_OVER_THIRD = 0.12

BLANK_MAX_SCORE = 0.34


# ============================================================
# BUBBLE CROPPING
# ============================================================

def crop_bubble(
    gray,
    x,
    y,
    radius=DEFAULT_CROP_RADIUS,
):
    h, w = gray.shape[:2]

    x1 = max(
        0,
        int(round(x - radius)),
    )

    y1 = max(
        0,
        int(round(y - radius)),
    )

    x2 = min(
        w,
        int(round(x + radius + 1)),
    )

    y2 = min(
        h,
        int(round(y + radius + 1)),
    )

    return gray[
        y1:y2,
        x1:x2,
    ]


# ============================================================
# CLASSICAL BUBBLE FEATURES
# ============================================================

def _circle_mask(
    size,
    radius,
):
    center = (
        size - 1
    ) / 2.0

    yy, xx = np.ogrid[
        :size,
        :size,
    ]

    return (
        (
            xx - center
        ) ** 2
        +
        (
            yy - center
        ) ** 2
        <= radius ** 2
    )


def _bubble_metrics(
    crop,
):
    """
    Measure fill characteristics while suppressing the printed outline.

    Core idea:
      - a true fill darkens the bubble CENTER
      - an empty printed outline mostly darkens the outer ring
    """

    if (
        crop is None
        or crop.size == 0
    ):
        return {
            "core_dark_ratio": 0.0,
            "disk_dark_ratio": 0.0,
            "core_mean": 255.0,
            "paper_mean": 255.0,
            "center_contrast": 0.0,
            "classical_score": 0.0,
        }

    if crop.ndim == 3:
        crop = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

    crop = crop.astype(
        np.uint8
    )

    # Mild normalization only.
    clahe = cv2.createCLAHE(
        clipLimit=1.8,
        tileGridSize=(
            4,
            4,
        ),
    )

    normalized = clahe.apply(
        crop
    )

    h, w = normalized.shape[:2]

    size = min(
        h,
        w,
    )

    y0 = (
        h - size
    ) // 2

    x0 = (
        w - size
    ) // 2

    square = normalized[
        y0:y0 + size,
        x0:x0 + size,
    ]

    core_radius = max(
        3.0,
        size * 0.18,
    )

    disk_radius = max(
        5.0,
        size * 0.31,
    )

    paper_inner_radius = (
        size * 0.39
    )

    paper_outer_radius = (
        size * 0.49
    )

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
        paper_outer_radius,
    )

    inner_outer_mask = _circle_mask(
        size,
        paper_inner_radius,
    )

    paper_mask = np.logical_and(
        outer_mask,
        np.logical_not(
            inner_outer_mask
        ),
    )

    core_pixels = square[
        core_mask
    ]

    disk_pixels = square[
        disk_mask
    ]

    paper_pixels = square[
        paper_mask
    ]

    if core_pixels.size == 0:
        core_pixels = square.reshape(
            -1
        )

    if disk_pixels.size == 0:
        disk_pixels = square.reshape(
            -1
        )

    if paper_pixels.size == 0:
        paper_pixels = square.reshape(
            -1
        )

    paper_mean = float(
        np.mean(
            paper_pixels
        )
    )

    core_mean = float(
        np.mean(
            core_pixels
        )
    )

    center_contrast = float(
        max(
            0.0,
            paper_mean - core_mean,
        )
    )

    # Dynamic local dark threshold.
    dark_threshold = int(
        np.clip(
            paper_mean - 38.0,
            85,
            160,
        )
    )

    core_dark_ratio = float(
        np.mean(
            core_pixels
            <
            dark_threshold
        )
    )

    disk_dark_ratio = float(
        np.mean(
            disk_pixels
            <
            dark_threshold
        )
    )

    # Combine classical features.
    contrast_score = float(
        np.clip(
            center_contrast
            /
            95.0,
            0.0,
            1.0,
        )
    )

    darkness_score = float(
        np.clip(
            (
                190.0 - core_mean
            )
            /
            110.0,
            0.0,
            1.0,
        )
    )

    classical_score = (
        0.50
        *
        core_dark_ratio
        +
        0.22
        *
        disk_dark_ratio
        +
        0.18
        *
        contrast_score
        +
        0.10
        *
        darkness_score
    )

    return {
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

        "center_contrast":
            round(
                center_contrast,
                2,
            ),

        "classical_score":
            round(
                float(
                    classical_score
                ),
                4,
            ),
    }


# ============================================================
# ML HELPERS
# ============================================================

def _ml_filled_probability(
    prediction,
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
        and "filled"
        in probabilities
    ):
        return float(
            probabilities[
                "filled"
            ]
        )

    label = str(
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

    if label == "filled":
        return confidence

    return 0.0


def _ml_ambiguous_probability(
    prediction,
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
        and "ambiguous"
        in probabilities
    ):
        return float(
            probabilities[
                "ambiguous"
            ]
        )

    label = str(
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

    if label == "ambiguous":
        return confidence

    return 0.0


# ============================================================
# RELATIVE QUESTION SCORING
# ============================================================

def _normalize_within_question(
    values,
):
    """
    Convert four raw values to a 0..1 relative scale.

    The lightest/least-filled bubble becomes near 0 and the strongest
    becomes near 1. This makes the detector more robust to lighting
    changes down the page.
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    minimum = float(
        np.min(
            values
        )
    )

    maximum = float(
        np.max(
            values
        )
    )

    spread = (
        maximum - minimum
    )

    if spread < 1e-6:
        return np.zeros_like(
            values,
            dtype=np.float32,
        )

    return (
        values - minimum
    ) / spread


def _build_question_scores(
    option_data,
):
    options = list(
        option_data.keys()
    )

    classical = np.array(
        [
            float(
                option_data[
                    option
                ][
                    "metrics"
                ][
                    "classical_score"
                ]
            )
            for option
            in options
        ],
        dtype=np.float32,
    )

    core_dark = np.array(
        [
            float(
                option_data[
                    option
                ][
                    "metrics"
                ][
                    "core_dark_ratio"
                ]
            )
            for option
            in options
        ],
        dtype=np.float32,
    )

    contrast = np.array(
        [
            float(
                option_data[
                    option
                ][
                    "metrics"
                ][
                    "center_contrast"
                ]
            )
            for option
            in options
        ],
        dtype=np.float32,
    )

    ml_fill = np.array(
        [
            float(
                option_data[
                    option
                ][
                    "ml_filled_probability"
                ]
            )
            for option
            in options
        ],
        dtype=np.float32,
    )

    rel_classical = _normalize_within_question(
        classical
    )

    rel_core = _normalize_within_question(
        core_dark
    )

    rel_contrast = _normalize_within_question(
        contrast
    )

    rel_ml = _normalize_within_question(
        ml_fill
    )

    result = {}

    for index, option in enumerate(
        options
    ):

        absolute_score = (
            0.62
            *
            classical[
                index
            ]
            +
            0.38
            *
            ml_fill[
                index
            ]
        )

        relative_score = (
            0.46
            *
            rel_classical[
                index
            ]
            +
            0.28
            *
            rel_core[
                index
            ]
            +
            0.16
            *
            rel_contrast[
                index
            ]
            +
            0.10
            *
            rel_ml[
                index
            ]
        )

        # Relative comparison gets most of the weight.
        final_score = (
            0.66
            *
            relative_score
            +
            0.34
            *
            absolute_score
        )

        result[
            option
        ] = {
            **option_data[
                option
            ],

            "absolute_score":
                round(
                    float(
                        absolute_score
                    ),
                    4,
                ),

            "relative_score":
                round(
                    float(
                        relative_score
                    ),
                    4,
                ),

            "final_score":
                round(
                    float(
                        final_score
                    ),
                    4,
                ),
        }

    return result


def _decide_question(
    scored,
):
    ranked = sorted(
        scored.items(),
        key=lambda item:
            item[1][
                "final_score"
            ],
        reverse=True,
    )

    best_option, best_info = (
        ranked[0]
    )

    second_option, second_info = (
        ranked[1]
    )

    third_option, third_info = (
        ranked[2]
    )

    fourth_option, fourth_info = (
        ranked[3]
    )

    best = float(
        best_info[
            "final_score"
        ]
    )

    second = float(
        second_info[
            "final_score"
        ]
    )

    third = float(
        third_info[
            "final_score"
        ]
    )

    top_gap = (
        best - second
    )

    second_third_gap = (
        second - third
    )

    best_absolute = float(
        best_info[
            "absolute_score"
        ]
    )

    second_absolute = float(
        second_info[
            "absolute_score"
        ]
    )

    # --------------------------------------------------------
    # BLANK
    # --------------------------------------------------------

    # If even the strongest bubble has weak absolute evidence,
    # do not force a relative winner.
    if (
        best_absolute
        <=
        BLANK_MAX_SCORE
        and
        best
        <
        0.58
    ):
        return {
            "answer":
                None,

            "status":
                "blank",

            "best_option":
                best_option,

            "best_score":
                best,

            "second_option":
                second_option,

            "second_score":
                second,

            "top_gap":
                top_gap,

            "second_third_gap":
                second_third_gap,
        }

    # --------------------------------------------------------
    # TRUE MULTIPLE
    # --------------------------------------------------------

    # Two options must BOTH be independently strong.
    # They must also be close to one another AND clearly separated
    # from option #3. This prevents empty outlines from creating MULTI.
    if (
        best
        >=
        MULTI_MIN_SCORE
        and
        second
        >=
        MULTI_MIN_SCORE
        and
        best_absolute
        >=
        0.42
        and
        second_absolute
        >=
        0.42
        and
        top_gap
        <=
        MULTI_MAX_GAP_BETWEEN_TOP_TWO
        and
        second_third_gap
        >=
        MULTI_MIN_GAP_OVER_THIRD
    ):
        return {
            "answer":
                "MULTIPLE",

            "status":
                "multiple",

            "multiple_options":
                [
                    best_option,
                    second_option,
                ],

            "best_option":
                best_option,

            "best_score":
                best,

            "second_option":
                second_option,

            "second_score":
                second,

            "top_gap":
                top_gap,

            "second_third_gap":
                second_third_gap,
        }

    # --------------------------------------------------------
    # SINGLE
    # --------------------------------------------------------

    if (
        best
        >=
        SINGLE_MIN_SCORE
        and
        top_gap
        >=
        SINGLE_MIN_GAP
    ):
        return {
            "answer":
                best_option,

            "status":
                "answered",

            "best_option":
                best_option,

            "best_score":
                best,

            "second_option":
                second_option,

            "second_score":
                second,

            "top_gap":
                top_gap,

            "second_third_gap":
                second_third_gap,
        }

    # --------------------------------------------------------
    # AMBIGUOUS FALLBACK
    # --------------------------------------------------------

    return {
        "answer":
            None,

        "status":
            "ambiguous",

        "best_option":
            best_option,

        "best_score":
            best,

        "second_option":
            second_option,

        "second_score":
            second,

        "top_gap":
            top_gap,

        "second_third_gap":
            second_third_gap,
    }


# ============================================================
# PUBLIC READER
# ============================================================

def scan_answers_ml(
    gray,
    coordinates,
    crop_radius=DEFAULT_CROP_RADIUS,
    filled_confidence=0.70,
    ambiguous_confidence=0.60,
):
    """
    Relative hybrid OMR reader.

    Strategy:
      1. crop A/B/C/D for every question
      2. run ML in one batch
      3. compute classical center-fill metrics
      4. compare A/B/C/D RELATIVE TO EACH OTHER
      5. return single / blank / multiple / ambiguous

    The key improvement is that brightness/shadow variation down the
    sheet is handled per-question instead of using only global thresholds.
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

    raw_question_data = {}

    for question, option_map in coordinates.items():

        raw_question_data[
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

            batch_crops.append(
                crop
            )

            batch_map.append(
                (
                    question,
                    option,
                )
            )

            raw_question_data[
                question
            ][
                option
            ] = {
                "metrics":
                    metrics,

                "crop_center": [
                    int(
                        round(
                            x
                        )
                    ),
                    int(
                        round(
                            y
                        )
                    ),
                ],
            }

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

        raw_question_data[
            question
        ][
            option
        ][
            "ml"
        ] = prediction

        raw_question_data[
            question
        ][
            option
        ][
            "ml_filled_probability"
        ] = round(
            _ml_filled_probability(
                prediction
            ),
            4,
        )

        raw_question_data[
            question
        ][
            option
        ][
            "ml_ambiguous_probability"
        ] = round(
            _ml_ambiguous_probability(
                prediction
            ),
            4,
        )

    answers = {}
    debug = {}

    for question, option_data in raw_question_data.items():

        scored = _build_question_scores(
            option_data
        )

        decision = _decide_question(
            scored
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

            "options":
                scored,
        }

    return (
        answers,
        debug,
    )
