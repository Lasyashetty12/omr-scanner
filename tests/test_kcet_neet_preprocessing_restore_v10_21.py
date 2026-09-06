
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_kcet_neet_answers_use_pre_adaptive_answer_image():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "prepare_neet_kcet_answer_image_v10_21("
        in source
    )

    assert (
        "neet_kcet_answer_image ="
        in source
    )

    assert (
        source.count(
            "scan_answers(\n"
            "                neet_kcet_answer_image,"
        )
        >= 2
    )


def test_answer_mode_excludes_gamma_and_saturation_recovery():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(
        encoding="utf-8"
    )

    start = source.index(
        "def prepare_neet_kcet_answer_image_v10_21("
    )

    end = source.index(
        "\n# ============================================================",
        start,
    )

    helper = source[start:end]

    assert "cv2.LUT" not in helper
    assert "cv2.cvtColor(" in helper
    assert "cv2.createCLAHE" in helper
    assert "cv2.bilateralFilter" in helper
    assert "cv2.addWeighted" in helper


def test_current_adaptive_document_mode_remains_for_other_paths():
    document_mode = (
        ROOT
        / "omr_preprocess"
        / "document_mode.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "adaptive_document_mode_v3"
        in document_mode
    )

    assert (
        "_adaptive_capture_enhancement("
        in document_mode
    )


def test_stable_fitted_kcet_neet_mapping_remains():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "_stable_neet_kcet_mapping_v10_20"
        in source
    )

    assert (
        "coordinates=fitted_coordinates"
        in source
    )


def test_jee_paths_untouched():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "jee_ml_hybrid_gate_v10_11"
        in source
    )

    assert (
        "merge_jee_camera_numerical_records("
        in source
    )

    assert (
        'template_exam_name in ("NEET", "KCET")'
        in source
    )


def test_new_answer_image_keeps_geometry():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(
        encoding="utf-8"
    )

    start = source.index(
        "def prepare_neet_kcet_answer_image_v10_21("
    )

    end = source.index(
        "\n# ============================================================",
        start,
    )

    helper = source[start:end]

    assert "cv2.resize" not in helper
    assert "warpPerspective" not in helper
    assert "getPerspectiveTransform" not in helper


def test_batch_limit_remains_500():
    source = (
        ROOT
        / "static"
        / "app.js"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "MAX_BATCH_OMR_FILES = 500"
        in source
    )
