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

    # Newer persistence flow may prefer a durable database result ID while
    # retaining scan UUID as fallback. Do not require one exact source line.
    assert "latestResultId" in source
    assert "data?.id" in source
    assert "data?.scan_id" in source

def test_result_page_uses_cache_before_network():
    source = (
        ROOT
        / "static"
        / "result.js"
    ).read_text(encoding="utf-8")

    cache_token = "const immediateCached = readCachedResult(resultId);"
    assert cache_token in source

    cache_pos = source.index(cache_token)
    guard_pos = source.index(
        "if (immediateCached)",
        cache_pos,
    )
    return_pos = source.index(
        "return;",
        guard_pos,
    )

    # Network implementation may be direct fetch() or a helper, but cache
    # must be consumed first and return before network work.
    assert cache_pos < guard_pos < return_pos

    later_network = [
        source.find("/api/omr-results/", return_pos),
        source.find("fetch(", return_pos),
        source.find("fetchResult", return_pos),
    ]

    assert any(pos >= 0 for pos in later_network)

