import inspect
from pathlib import Path

import scanner


def test_json_anchored_reader_remains_import_compatible():
    signature = inspect.signature(
        scanner.scan_answers_json_anchored
    )

    accepted = set(signature.parameters)
    assert {
        "gray",
        "template",
        "fitted_coordinates",
        "crop_radius",
    }.issubset(accepted)

    source = Path(scanner.__file__).read_text(
        encoding="utf-8"
    )

    # v10.20 keeps the JSON reader available for compatibility/identity work,
    # but calibrated fitted coordinates are again primary for answer mapping.
    assert "_stable_neet_kcet_mapping_v10_20" in source
    assert "coordinates=fitted_coordinates" in source


def test_generated_kcet_template_tests_are_present():
    root = Path(scanner.__file__).resolve().parent

    assert (
        root
        / "tests"
        / "test_generated_omr_templates.py"
    ).exists()
