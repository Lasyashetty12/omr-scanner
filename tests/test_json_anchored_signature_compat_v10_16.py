import inspect
from pathlib import Path

import scanner


def test_scanner_json_anchored_call_matches_reader_signature():
    signature = inspect.signature(
        scanner.scan_answers_json_anchored
    )

    accepted = set(signature.parameters)

    source = Path(scanner.__file__).read_text(
        encoding="utf-8"
    )

    call_start = source.index(
        "scan_answers_json_anchored("
    )

    call_end = source.index(
        "\n        )\n",
        call_start,
    )

    call_source = source[
        call_start:
        call_end
    ]

    for keyword in (
        "filled_confidence",
        "ambiguous_confidence",
        "questions_per_column",
    ):
        if keyword not in accepted:
            assert (keyword + "=") not in call_source


def test_generated_kcet_template_tests_are_present():
    root = Path(scanner.__file__).resolve().parent

    assert (
        root
        / "tests"
        / "test_generated_omr_templates.py"
    ).exists()
