from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_jee_numeric_v2_uses_real_grid_calibration():
    source = (
        ROOT
        / "jee_precise_reader.py"
    ).read_text(encoding="utf-8")

    assert (
        "from jee_reader import _calibrate_numerical_question"
        in source
    )

    start = source.index(
        "def scan_jee_numerical_precise("
    )

    body = source[start:]

    assert "_calibrate_numerical_question(" in body
    assert 'debug["grid_cluster"] = grid_debug' in body


def test_jee_numeric_v2_selects_one_decimal_only():
    source = (
        ROOT
        / "jee_precise_reader.py"
    ).read_text(encoding="utf-8")

    start = source.index(
        "def _read_numeric_question("
    )

    end = source.index(
        "\ndef scan_jee_numerical_precise(",
        start,
    )

    body = source[start:end]

    assert "decimal_ranked" in body
    assert "selected_decimal" in body
    assert "decimal_candidates" not in body
    assert 'reader": "jee_precise_circle_reader_v2"' in body


def test_jee_numeric_overlay_draws_detected_v5_bubbles():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(encoding="utf-8")

    start = source.index(
        "def draw_jee_answer_analysis("
    )

    end = source.index(
        "\n\n# ============================================================\n# DEBUG IMAGE",
        start,
    )

    body = source[start:end]

    assert "filled_candidates" in body
    assert "filled_decimal_points" in body
    assert "decimal_color" in body
    assert "Numerical bubble confidence is per bubble" in body
