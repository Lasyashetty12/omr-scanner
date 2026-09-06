from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import cv2
import numpy as np


PROFILE = "jee_ml_vote_v10_32"


def _probability(
    option_or_prediction: Dict[str, Any],
    label: str,
) -> float:
    direct = "ml_" + label + "_probability"

    if direct in option_or_prediction:
        try:
            return float(
                option_or_prediction.get(
                    direct,
                    0.0,
                )
            )
        except (TypeError, ValueError):
            return 0.0

    prediction = option_or_prediction.get(
        "ml",
        option_or_prediction,
    )

    if not isinstance(prediction, dict):
        return 0.0

    probabilities = prediction.get(
        "probabilities",
        {},
    )

    if isinstance(probabilities, dict):
        try:
            return float(
                probabilities.get(
                    label,
                    0.0,
                )
            )
        except (TypeError, ValueError):
            return 0.0

    predicted_label = str(
        prediction.get(
            "label",
            "",
        )
    ).strip().lower()

    if predicted_label != label:
        return 0.0

    try:
        return float(
            prediction.get(
                "confidence",
                0.0,
            )
        )
    except (TypeError, ValueError):
        return 0.0


def _crop(
    gray: np.ndarray,
    center: Iterable[float],
    radius: int,
):
    center = list(center)

    if len(center) < 2:
        return None

    x = int(
        round(
            float(center[0])
        )
    )
    y = int(
        round(
            float(center[1])
        )
    )
    radius = int(radius)

    x0 = x - radius
    y0 = y - radius
    x1 = x + radius + 1
    y1 = y + radius + 1

    if (
        x0 < 0
        or y0 < 0
        or x1 > gray.shape[1]
        or y1 > gray.shape[0]
    ):
        return None

    crop = gray[
        y0:y1,
        x0:x1,
    ]

    if crop.size == 0:
        return None

    return crop.copy()


def _aggregate(
    views,
):
    """
    Conservative per-bubble ml_omr voting.

    A single crop may be wrong. A bubble is accepted as FILLED only when
    at least 3 independent crop scales vote filled. A bubble is decisively
    BLANK only when at least 3 views strongly vote blank.
    """
    filled = np.asarray(
        [float(view["filled"]) for view in views],
        dtype=np.float32,
    )
    blank = np.asarray(
        [float(view["blank"]) for view in views],
        dtype=np.float32,
    )
    ambiguous = np.asarray(
        [float(view["ambiguous"]) for view in views],
        dtype=np.float32,
    )

    fill_margin = filled - blank
    blank_margin = blank - filled

    fill_votes = int(
        np.count_nonzero(
            (filled >= 0.58)
            & (fill_margin >= 0.10)
        )
    )

    strong_fill_votes = int(
        np.count_nonzero(
            (filled >= 0.72)
            & (fill_margin >= 0.20)
        )
    )

    blank_votes = int(
        np.count_nonzero(
            (blank >= 0.70)
            & (blank_margin >= 0.20)
        )
    )

    strong_blank_votes = int(
        np.count_nonzero(
            (blank >= 0.82)
            & (blank_margin >= 0.35)
        )
    )

    median_filled = float(np.median(filled))
    median_blank = float(np.median(blank))
    median_ambiguous = float(np.median(ambiguous))
    median_fill_margin = float(np.median(fill_margin))
    max_filled = float(np.max(filled))

    is_filled = bool(
        len(views) >= 4
        and fill_votes >= 3
        and strong_fill_votes >= 2
        and median_filled >= 0.60
        and median_fill_margin >= 0.12
        and median_blank <= 0.30
    )

    is_decisive_blank = bool(
        len(views) >= 4
        and blank_votes >= 3
        and strong_blank_votes >= 2
        and median_blank >= 0.78
        and median_filled <= 0.22
        and max_filled <= 0.45
    )

    return {
        "views": views,
        "fill_votes": fill_votes,
        "strong_fill_votes": strong_fill_votes,
        "blank_votes": blank_votes,
        "strong_blank_votes": strong_blank_votes,
        "median_filled": round(median_filled, 4),
        "median_blank": round(median_blank, 4),
        "median_ambiguous": round(median_ambiguous, 4),
        "median_fill_margin": round(median_fill_margin, 4),
        "max_filled": round(max_filled, 4),
        "is_filled": is_filled,
        "is_decisive_blank": is_decisive_blank,
    }


def _stable_answer(
    stable_mcq: Dict[Any, Any],
    question: int,
) -> str:
    if not isinstance(stable_mcq, dict):
        return ""

    record = stable_mcq.get(
        question,
        stable_mcq.get(
            str(question),
            {},
        ),
    )

    if not isinstance(record, dict):
        return ""

    return str(
        record.get(
            "answer",
            "",
        )
        or ""
    ).strip().upper()


def refine_jee_multiscale_multiples(
    *,
    gray: np.ndarray,
    stable_mcq: Dict[Any, Any],
    ml_answers: Dict[Any, Any],
    ml_debug: Dict[Any, Any],
    radii: Tuple[int, ...] = (
        12,
        14,
        16,
    ),
    _classifier=None,
):
    """
    JEE-MCQ-only second-pass using the EXISTING ml_omr ONNX model.

    The normal JEE path supplies one tight radius-10 crop. That can miss a
    genuine second mark when the crop is slightly tight/off-centre. We keep
    that existing prediction and add radius 12/14/16 views around the SAME
    CV-fitted physical centre.

    KCET/NEET do not call this module.
    """
    if gray is None:
        return ml_answers, ml_debug

    if not isinstance(ml_debug, dict):
        return ml_answers, ml_debug

    if gray.ndim == 3:
        gray = cv2.cvtColor(
            gray,
            cv2.COLOR_BGR2GRAY,
        )

    if _classifier is None:
        from ml_omr.inference import classify_batch
        classifier = classify_batch
    else:
        classifier = _classifier

    options = (
        "A",
        "B",
        "C",
        "D",
    )

    final_answers = dict(
        ml_answers
        if isinstance(ml_answers, dict)
        else {}
    )
    final_debug = dict(ml_debug)

    evidence = {}
    samples = []
    sample_keys = []

    for question_key, decision in ml_debug.items():
        try:
            question = int(
                question_key
            )
        except (TypeError, ValueError):
            continue

        if not isinstance(decision, dict):
            continue

        option_data = decision.get(
            "options",
            {},
        )

        if not isinstance(option_data, dict):
            continue

        evidence[question] = {}

        for option in options:
            info = option_data.get(
                option,
                {},
            )

            if not isinstance(info, dict):
                continue

            center = info.get(
                "crop_center"
            )

            if (
                not isinstance(
                    center,
                    (list, tuple),
                )
                or len(center) < 2
            ):
                continue

            evidence[
                question
            ][
                option
            ] = [
                {
                    "radius": 10,
                    "filled": round(
                        _probability(
                            info,
                            "filled",
                        ),
                        6,
                    ),
                    "blank": round(
                        _probability(
                            info,
                            "blank",
                        ),
                        6,
                    ),
                    "ambiguous": round(
                        _probability(
                            info,
                            "ambiguous",
                        ),
                        6,
                    ),
                    "source":
                        "existing_hybrid_ml",
                }
            ]

            for radius in radii:
                crop = _crop(
                    gray,
                    center,
                    int(radius),
                )

                if crop is None:
                    continue

                samples.append(
                    crop
                )
                sample_keys.append(
                    (
                        question,
                        option,
                        int(radius),
                    )
                )

    predictions = []

    try:
        # Keep each ONNX batch no larger than the ordinary JEE/OMR option
        # batch, avoiding an unnecessary large model call.
        for start in range(
            0,
            len(samples),
            240,
        ):
            batch = samples[
                start:
                start + 240
            ]

            if not batch:
                continue

            result = classifier(
                batch
            )

            if len(result) != len(batch):
                raise ValueError(
                    "Unexpected ml_omr batch size."
                )

            predictions.extend(
                result
            )

    except Exception as error:
        # Rescue pass only: never degrade the established JEE answer if the
        # extra model pass cannot run.
        final_debug[
            "_jee_multiscale_ml_error"
        ] = str(error)

        return (
            final_answers,
            final_debug,
        )

    for key, prediction in zip(
        sample_keys,
        predictions,
    ):
        (
            question,
            option,
            radius,
        ) = key

        evidence[
            question
        ][
            option
        ].append(
            {
                "radius":
                    int(radius),
                "filled":
                    round(
                        _probability(
                            prediction,
                            "filled",
                        ),
                        6,
                    ),
                "blank":
                    round(
                        _probability(
                            prediction,
                            "blank",
                        ),
                        6,
                    ),
                "ambiguous":
                    round(
                        _probability(
                            prediction,
                            "ambiguous",
                        ),
                        6,
                    ),
                "source":
                    "direct_multiscale_ml",
            }
        )

    for question in sorted(
        evidence
    ):
        if len(
            evidence[
                question
            ]
        ) != 4:
            continue

        summary = {
            option:
                _aggregate(
                    evidence[
                        question
                    ][
                        option
                    ]
                )
            for option
            in options
        }

        stable = _stable_answer(
            stable_mcq,
            question,
        )

        filled_options = [
            option
            for option in options
            if summary[
                option
            ][
                "is_filled"
            ]
        ]

        decisive_blank_options = [
            option
            for option in options
            if summary[
                option
            ][
                "is_decisive_blank"
            ]
        ]

        decision = final_debug.get(
            question,
            final_debug.get(
                str(question),
                {},
            ),
        )

        if not isinstance(decision, dict):
            decision = {}

        updated = dict(decision)
        updated[
            "jee_multiscale_ml_profile"
        ] = PROFILE
        updated[
            "jee_multiscale_ml_options"
        ] = summary
        updated[
            "jee_ml_filled_options"
        ] = list(filled_options)
        updated[
            "jee_ml_decisive_blank_options"
        ] = list(
            decisive_blank_options
        )

        if stable in options:
            secondary = [
                option
                for option in filled_options
                if option != stable
            ]

            # Exactly one secondary fill. Never invent a 3/4-option MULTIPLE.
            if (
                len(secondary) == 1
                and len(filled_options) <= 2
                and not summary[
                    stable
                ][
                    "is_decisive_blank"
                ]
            ):
                multiple_options = [
                    stable,
                    secondary[0],
                ]

                updated[
                    "answer"
                ] = "MULTIPLE"
                updated[
                    "status"
                ] = "multiple"
                updated[
                    "best_option"
                ] = stable
                updated[
                    "multiple_options"
                ] = multiple_options
                updated[
                    "jee_multiscale_ml_multiple"
                ] = True
                updated[
                    "jee_ml_vote_multiple"
                ] = True

                final_answers[
                    question
                ] = "MULTIPLE"

            # Blank-as-filled correction is intentionally very strict:
            # all A/B/C/D must independently look blank to ml_omr.
            elif (
                len(filled_options) == 0
                and len(
                    decisive_blank_options
                ) == 4
            ):
                updated[
                    "answer"
                ] = None
                updated[
                    "status"
                ] = "blank"
                updated[
                    "multiple_options"
                ] = []
                updated[
                    "jee_multiscale_ml_blank_veto"
                ] = True
                updated[
                    "jee_ml_vote_blank"
                ] = True

                final_answers[
                    question
                ] = None

        else:
            # Unstable/blank row becomes MULTIPLE only when exactly two
            # physical bubble positions are independently ML-filled.
            if len(filled_options) == 2:
                updated[
                    "answer"
                ] = "MULTIPLE"
                updated[
                    "status"
                ] = "multiple"
                updated[
                    "best_option"
                ] = filled_options[0]
                updated[
                    "multiple_options"
                ] = list(
                    filled_options
                )
                updated[
                    "jee_multiscale_ml_multiple"
                ] = True
                updated[
                    "jee_ml_vote_multiple"
                ] = True

                final_answers[
                    question
                ] = "MULTIPLE"

        final_debug[
            question
        ] = updated


    return (
        final_answers,
        final_debug,
    )
