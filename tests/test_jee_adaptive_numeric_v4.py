from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v4_adaptive_reader_has_been_replaced_by_v5():
    source = (
        ROOT
        / "jee_reader.py"
    ).read_text(encoding="utf-8")

    assert "def _solid_core_metrics(" in source
    assert "filled_candidates" in source
    assert "jee_solid_core_reader_v5" in source


def test_v5_debug_overlay_is_per_detected_bubble():
    source = (
        ROOT
        / "scanner.py"
    ).read_text(encoding="utf-8")

    assert "filled_candidates" in source
    assert "filled_decimal_points" in source
    assert (
        "Numerical bubble confidence is per bubble"
        in source
    )
