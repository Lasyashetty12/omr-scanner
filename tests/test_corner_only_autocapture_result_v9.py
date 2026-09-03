from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_captured_preview_is_not_replaced_by_internal_canonical_image():
    source = (
        ROOT
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "Keep the exact camera/upload preview visible after evaluation." in source

    display_start = source.index(
        "function displayResult("
    )
    scan_start = source.index(
        "async function scanOMR()",
        display_start,
    )
    display_block = source[
        display_start:scan_start
    ]

    assert "correctedUrl" not in display_block


def test_result_link_prefers_scan_uuid():
    source = (
        ROOT
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "latestResultId = data?.scan_id || data?.id || null;" in source
    assert 'data-id="${r.scan_id || r.id}"' in source


def test_result_page_uses_cache_before_network():
    source = (
        ROOT
        / "static"
        / "result.js"
    ).read_text(encoding="utf-8")

    cache_pos = source.index(
        "const immediateCached = readCachedResult(resultId);"
    )
    fetch_pos = source.index(
        "await fetch(`/api/omr-results/"
    )

    assert cache_pos < fetch_pos
    assert "displayResultData(immediateCached);" in source
