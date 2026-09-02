from pathlib import Path
import json
import cv2

ROOT = Path(__file__).resolve().parents[1]


def test_blank_reference_sheet_remains_blank_under_v5():
    from jee_reader import (
        scan_jee_numerical_sections_robust,
    )

    template = json.loads(
        (
            ROOT
            / "templates"
            / "jee.json"
        ).read_text(encoding="utf-8")
    )

    image = cv2.imread(
        str(
            ROOT
            / "references"
            / template["reference_image"]
        )
    )

    assert image is not None

    detected, _debug = (
        scan_jee_numerical_sections_robust(
            image,
            template,
        )
    )

    wrong = {
        q: record["answer"]
        for q, record in detected.items()
        if record["answer"] != "BLANK"
    }

    assert wrong == {}


def test_v3_reference_reader_has_been_replaced_by_v5():
    source = (
        ROOT
        / "jee_reader.py"
    ).read_text(encoding="utf-8")

    assert "def _solid_core_metrics(" in source
    assert "jee_solid_core_column_v5" in source
    assert "jee_solid_core_reader_v5" in source
