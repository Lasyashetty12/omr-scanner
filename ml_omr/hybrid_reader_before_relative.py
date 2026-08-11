from __future__ import annotations

import cv2
import numpy as np

from ml_omr.inference import classify_batch


# ============================================================
# BUBBLE CROP
# ============================================================

def crop_bubble(
    gray,
    x,
    y,
    radius=16,
):
    h, w = gray.shape[:2]

    x1 = max(
        0,
        int(x - radius),
    )
    y1 = max(
        0,
        int(y - radius),
    )
    x2 = min(
        w,
        int(x + radius + 1),
    )
    y2 = min(
        h,
        int(y + radius + 1),
    )

    return gray[
        y1:y2,
        x1:x2,
    ]


# ============================================================
# CLASSICAL FILL VERIFICATION
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
    Measure how much of the INNER bubble is truly dark.

    Printed empty bubble outlines mainly live near the outer ring.
    A genuinely filled bubble darkens the centre as well.

    Returns normalized metrics used to verify the ML prediction.
    """

    if crop is None or crop.size == 0:
        return {
            "core_dark_ratio": 0.0,
            "disk_dark_ratio": 0.0,
            "core_mean": 255.0,
            "background_mean": 255.0,
            "center_contrast": 0.0,
            "fill_strength": 0.0,
        }

    if crop.ndim == 3:
        crop = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

    crop = crop.astype(
        np.uint8
    )

    # Normalize only gently. We do NOT threshold the full image.
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

    if size < 9:
        return {
            "core_dark_ratio": 0.0,
            "disk_dark_ratio": 0.0,
            "core_mean": 255.0,
            "background_mean": 255.0,
            "center_contrast": 0.0,
            "fill_strength": 0.0,
        }

    # Center-crop to a square.
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

    # The ML crop is usually radius 16 => 33x33.
    # Use only the central part of the bubble for fill confirmation.
    core_radius = max(
        3.0,
        size * 0.18,
    )

    disk_radius = max(
        5.0,
        size * 0.31,
    )

    background_inner = (
        size * 0.37
    )

    background_outer = (
        size * 0.48
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

    core_pixels = square[
        core_mask
    ]

    disk_pixels = square[
        disk_mask
    ]

    background_pixels = square[
        background_mask
    ]

    if core_pixels.size == 0:
        core_pixels = square.reshape(
            -1
        )

    if disk_pixels.size == 0:
        disk_pixels = square.reshape(
            -1
        )

    if background_pixels.size == 0:
        background_pixels = square.reshape(
            -1
        )

    # Dynamic threshold derived from local paper tone.
    background_mean = float(
        np.mean(
            background_pixels
        )
    )

    dark_threshold = int(
        np.clip(
            background_mean - 38.0,
            85,
            155,
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

    core_mean = float(
        np.mean(
            core_pixels
        )
    )

    center_contrast = float(
        max(
            0.0,
            background_mean
            -
            core_mean,
        )
    )

    # 0..1-ish score.
    # Centre darkness dominates because an empty printed outline
    # should not have a dark centre.
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
                190.0
                -
                core_mean
            )
            /
            105.0,
            0.0,
            1.0,
        )
    )

    fill_strength = (
        0.48
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
        0.12
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

        "background_mean":
            round(
                background_mean,
                2,
            ),

        "center_contrast":
            round(
                center_contrast,
                2,
            ),

        "fill_strength":
            round(
                float(
                    fill_strength
                ),
                4,
            ),
    }


def _classical_state(
    metrics,
):
    """
    Conservative verifier.

    The most important rule:
    an ML "filled" prediction is NOT trusted when the bubble centre
    looks like an empty printed bubble.
    """

    core = float(
        metrics[
            "core_dark_ratio"
        ]
    )

    disk = float(
        metrics[
            "disk_dark_ratio"
        ]
    )

    core_mean = float(
        metrics[
            "core_mean"
        ]
    )

    contrast = float(
        metrics[
            "center_contrast"
        ]
    )

    strength = float(
        metrics[
            "fill_strength"
        ]
    )

    # Strong actual fill.
    if (
        core >= 0.52
        or (
            core >= 0.40
            and disk >= 0.34
        )
        or (
            core_mean <= 118
            and contrast >= 35
        )
        or strength >= 0.52
    ):
        return "filled"

    # Strong blank / printed-outline pattern.
    if (
        core <= 0.17
        and disk <= 0.28
        and core_mean >= 142
        and strength <= 0.30
    ):
        return "blank"

    return "uncertain"


# ============================================================
# HYBRID READER
# ============================================================

def scan_answers_ml(
    gray,
    coordinates,
    crop_radius=16,
    filled_confidence=0.70,
    ambiguous_confidence=0.60,
):
    """
    Hybrid OMR reader.

    ML is still used, but deterministic centre-fill verification prevents
    empty printed bubble outlines from being accepted as filled.

    coordinates:
        {
            1: {
                "A": (x, y),
                "B": (x, y),
                "C": (x, y),
                "D": (x, y),
            },
            ...
        }

    Returns:
        answers:
            {1: "A", 2: None, 3: "MULTIPLE", ...}

        debug:
            per-question ML + classical metrics
    """

    if gray.ndim == 3:
        gray = cv2.cvtColor(
            gray,
            cv2.COLOR_BGR2GRAY,
        )

    answers = {}
    debug = {}

    # --------------------------------------------------------
    # Build one large batch for ML inference
    # --------------------------------------------------------

    batch_crops = []
    batch_map = []
    metric_map = {}

    for question, option_map in coordinates.items():

        metric_map[
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

            batch_crops.append(
                crop
            )

            batch_map.append(
                (
                    question,
                    option,
                )
            )

            metrics = _bubble_metrics(
                crop
            )

            metric_map[
                question
            ][
                option
            ] = {
                **metrics,

                "classical_state":
                    _classical_state(
                        metrics
                    ),
            }

    predictions = classify_batch(
        batch_crops
    )

    grouped = {}

    for (
        question,
        option,
    ), prediction in zip(
        batch_map,
        predictions,
    ):

        grouped.setdefault(
            question,
            {}
        )[
            option
        ] = prediction

    # --------------------------------------------------------
    # Resolve each question
    # --------------------------------------------------------

    for question, option_predictions in grouped.items():

        option_metrics = metric_map[
            question
        ]

        # Combine ML with classical evidence.
        combined = {}

        for option, prediction in option_predictions.items():

            metrics = option_metrics[
                option
            ]

            classical_state = metrics[
                "classical_state"
            ]

            ml_label = str(
                prediction.get(
                    "label",
                    ""
                )
            ).lower()

            ml_confidence = float(
                prediction.get(
                    "confidence",
                    0.0,
                )
            )

            probabilities = prediction.get(
                "probabilities",
                {}
            )

            ml_filled_probability = float(
                probabilities.get(
                    "filled",
                    (
                        ml_confidence
                        if ml_label
                        ==
                        "filled"
                        else 0.0
                    ),
                )
            )

            strength = float(
                metrics[
                    "fill_strength"
                ]
            )

            # Classical verification dominates.
            if classical_state == "filled":
                hybrid_score = (
                    0.74
                    +
                    0.18
                    *
                    strength
                    +
                    0.08
                    *
                    ml_filled_probability
                )

            elif classical_state == "blank":
                # Even if ML says filled, cap it low.
                hybrid_score = (
                    0.08
                    *
                    ml_filled_probability
                    +
                    0.12
                    *
                    strength
                )

            else:
                # Uncertain classical region:
                # use ML as supporting evidence, not sole evidence.
                hybrid_score = (
                    0.56
                    *
                    strength
                    +
                    0.44
                    *
                    ml_filled_probability
                )

            combined[
                option
            ] = {
                "hybrid_score":
                    round(
                        float(
                            hybrid_score
                        ),
                        4,
                    ),

                "ml_label":
                    ml_label,

                "ml_confidence":
                    round(
                        ml_confidence,
                        4,
                    ),

                "ml_filled_probability":
                    round(
                        ml_filled_probability,
                        4,
                    ),

                "classical":
                    metrics,

                "ml":
                    prediction,
            }

        ranked = sorted(
            combined.items(),
            key=lambda item:
                item[1][
                    "hybrid_score"
                ],
            reverse=True,
        )

        if not ranked:
            answers[
                question
            ] = None

            debug[
                question
            ] = {
                "status":
                    "blank",

                "options":
                    {},
            }

            continue

        best_option, best_info = (
            ranked[0]
        )

        second_option, second_info = (
            ranked[1]
            if len(
                ranked
            ) > 1
            else (
                None,
                {
                    "hybrid_score":
                        0.0
                },
            )
        )

        best_score = float(
            best_info[
                "hybrid_score"
            ]
        )

        second_score = float(
            second_info[
                "hybrid_score"
            ]
        )

        gap = (
            best_score
            -
            second_score
        )

        strong_filled = [
            option
            for option, info
            in combined.items()
            if (
                info[
                    "classical"
                ][
                    "classical_state"
                ]
                ==
                "filled"
                and
                info[
                    "hybrid_score"
                ]
                >=
                0.74
            )
        ]

        plausible_filled = [
            option
            for option, info
            in combined.items()
            if (
                info[
                    "hybrid_score"
                ]
                >=
                0.54
                and
                info[
                    "classical"
                ][
                    "classical_state"
                ]
                !=
                "blank"
            )
        ]

        # ----------------------------------------------------
        # Decision rules
        # ----------------------------------------------------

        if len(
            strong_filled
        ) >= 2:

            # True multiple requires TWO independently strong,
            # centre-dark fills.
            answer = "MULTIPLE"
            status = "multiple"

        elif len(
            strong_filled
        ) == 1:

            answer = strong_filled[
                0
            ]
            status = "answered"

        elif (
            best_score
            >= 0.58
            and gap
            >= 0.10
            and best_info[
                "classical"
            ][
                "classical_state"
            ]
            !=
            "blank"
        ):

            answer = best_option
            status = "answered"

        elif (
            len(
                plausible_filled
            )
            == 1
            and best_info[
                "ml_label"
            ]
            ==
            "filled"
            and best_info[
                "ml_confidence"
            ]
            >=
            filled_confidence
        ):

            answer = plausible_filled[
                0
            ]
            status = "answered"

        else:

            # If all bubbles look blank, call it blank.
            all_blank = all(
                info[
                    "classical"
                ][
                    "classical_state"
                ]
                ==
                "blank"
                for info
                in combined.values()
            )

            # Preserve an ambiguous indication only when there is
            # some real centre-fill evidence.
            has_ambiguous_ml = any(
                info[
                    "ml_label"
                ]
                ==
                "ambiguous"
                and info[
                    "ml_confidence"
                ]
                >=
                ambiguous_confidence
                and info[
                    "classical"
                ][
                    "classical_state"
                ]
                !=
                "blank"
                for info
                in combined.values()
            )

            answer = None

            if all_blank:
                status = "blank"

            elif has_ambiguous_ml:
                status = "ambiguous"

            else:
                status = "blank"

        answers[
            question
        ] = answer

        debug[
            question
        ] = {
            "status":
                status,

            "best_option":
                best_option,

            "best_score":
                round(
                    best_score,
                    4,
                ),

            "second_option":
                second_option,

            "second_score":
                round(
                    second_score,
                    4,
                ),

            "confidence_gap":
                round(
                    gap,
                    4,
                ),

            "strong_filled_options":
                strong_filled,

            "plausible_filled_options":
                plausible_filled,

            "options":
                combined,
        }

    return (
        answers,
        debug,
    )
