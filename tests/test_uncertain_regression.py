import numpy as np

from ml_omr.hybrid_reader import _decide_question, _postprocess_known_failure_classes


def _option(darkness, disk, core=0.65):
    return {
        "metrics": {
            "center_darkness": darkness,
            "disk_dark_ratio": disk,
            "core_dark_ratio": core,
        },
        "ml_filled_probability": 0.10,
        "micro_core_darkness": 100.0,
        "crop_center": [100, 100],
    }


def test_unique_visual_winner_does_not_become_uncertain_for_low_ml_confidence():
    # A readable, uniquely dominant mark below the reader's stricter
    # strong/medium/faint thresholds must not be classified UNCERTAIN just
    # because the model confidence is low.
    options = {
        "A": _option(82.0, 0.50),
        "B": _option(65.0, 0.38),
        "C": _option(61.0, 0.30),
        "D": _option(58.0, 0.27),
    }
    decision = _decide_question(options, {})
    assert decision["status"] == "ambiguous"

    result = _postprocess_known_failure_classes(
        1,
        options,
        decision,
        np.full((220, 220), 255, dtype=np.uint8),
    )

    assert result["status"] == "answered"
    assert result["answer"] == "A"
    assert result["unique_visual_winner_rescue"]


def test_existing_multiple_decision_is_unchanged_by_uncertain_rescue():
    options = {
        "A": _option(110.0, 0.95, 0.95),
        "B": _option(105.0, 0.90, 0.92),
        "C": _option(48.0, 0.22),
        "D": _option(45.0, 0.20),
    }
    decision = _decide_question(options, {})

    assert decision["status"] == "multiple"
    result = _postprocess_known_failure_classes(
        1,
        options,
        decision,
        np.full((220, 220), 255, dtype=np.uint8),
    )
    assert result["status"] == "multiple"
    assert result["answer"] == "MULTIPLE"
