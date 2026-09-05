
from pathlib import Path
import numpy as np
import identity_reader
import scanner

ROOT = Path(__file__).resolve().parents[1]


def _asymmetric_reference():
    image = np.full(
        (600, 400, 3),
        255,
        dtype=np.uint8,
    )
    image[30:70, 90:310] = 0
    image[120:500, 25:120] = 220
    image[150:160, 35:105] = 0
    image[250:260, 35:105] = 0
    image[350:360, 35:105] = 0
    image[550:558, 80:340] = 0
    return image


def test_upright_reference_stays_upright(monkeypatch):
    reference = _asymmetric_reference()

    monkeypatch.setattr(
        scanner.cv2,
        "imread",
        lambda *_args, **_kwargs:
            reference.copy(),
    )

    corrected, debug = (
        scanner.ensure_neet_kcet_upright(
            reference.copy(),
            "dummy.png",
        )
    )

    assert debug["rotation_degrees"] == 0
    assert np.array_equal(corrected, reference)


def test_upside_down_reference_rotates_back(monkeypatch):
    reference = _asymmetric_reference()

    monkeypatch.setattr(
        scanner.cv2,
        "imread",
        lambda *_args, **_kwargs:
            reference.copy(),
    )

    upside_down = scanner.cv2.rotate(
        reference,
        scanner.cv2.ROTATE_180,
    )

    corrected, debug = (
        scanner.ensure_neet_kcet_upright(
            upside_down,
            "dummy.png",
        )
    )

    assert debug["rotation_degrees"] == 180
    assert np.array_equal(corrected, reference)


def test_roll_ml_disk_fallback(monkeypatch):
    gray = np.full(
        (700, 500),
        255,
        dtype=np.uint8,
    )

    config = {
        "x_positions":
            [80, 120, 160, 200, 240, 280, 320],
        "y_positions":
            [100, 135, 170, 205, 240, 275, 310, 345, 380, 415],
        "values":
            [str(i) for i in range(10)],
        "hough_margin":
            20,
        "max_calibration_delta":
            22,
        "ml_crop_radius":
            10,
    }

    expected = "2034167"

    for column, digit in enumerate(expected):
        x = config["x_positions"][column]
        y = config["y_positions"][int(digit)]

        yy, xx = np.ogrid[
            :gray.shape[0],
            :gray.shape[1],
        ]

        mask = (
            (xx - x) ** 2
            + (yy - y) ** 2
            <= 7 ** 2
        )
        gray[mask] = 0

    monkeypatch.setattr(
        identity_reader,
        "_hough",
        lambda *_args, **_kwargs: [],
    )

    import ml_omr.inference

    def fake_classify(crops):
        results = []
        for crop in crops:
            dark = float(np.mean(crop < 120))
            filled = 0.92 if dark > 0.20 else 0.04
            results.append(
                {
                    "label":
                        "filled" if filled > 0.5 else "blank",
                    "confidence":
                        max(filled, 1.0 - filled),
                    "probabilities":
                        {
                            "ambiguous": 0.02,
                            "blank": 1.0 - filled,
                            "filled": filled,
                        },
                }
            )
        return results

    monkeypatch.setattr(
        ml_omr.inference,
        "classify_batch",
        fake_classify,
    )

    result = (
        identity_reader._detect_roll_number_ml_fallback(
            gray,
            config,
            {"bubble_radius": 10},
        )
    )

    assert result["complete"] is True
    assert result["value"] == expected
    assert result["reader"] == "jee_roll_ml_disk_v10_14"


def test_scanner_has_corrected_jee_identity_retry():
    source = (ROOT / "scanner.py").read_text(
        encoding="utf-8"
    )
    assert "jee_identity_corrected_retry_v10_14" in source


def test_orientation_only_neet_kcet():
    source = (ROOT / "scanner.py").read_text(
        encoding="utf-8"
    )
    assert 'template_exam_name in ("NEET", "KCET")' in source
    assert "ensure_neet_kcet_upright(" in source


def test_existing_jee_answer_paths_untouched():
    source = (ROOT / "scanner.py").read_text(
        encoding="utf-8"
    )
    assert "jee_ml_hybrid_gate_v10_11" in source
    assert "merge_jee_camera_numerical_records(" in source


def test_batch_limit_500():
    source = (
        ROOT
        / "static"
        / "app.js"
    ).read_text(
        encoding="utf-8"
    )
    assert "MAX_BATCH_OMR_FILES = 500" in source
