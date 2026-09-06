from ml_omr.jee_solid_profile import _cluster_from_values


def cluster(values, darkening):
    return _cluster_from_values(
        values,
        darkening,
        split_gap=12.0,
        minimum_column_darkening=9.0,
    )["candidate_options"]


def test_two_real_fills_are_one_dark_cluster():
    assert set(
        cluster(
            {"A": 99, "B": 54, "C": 96, "D": 73},
            {"A": -4, "B": 35, "C": -2, "D": 20},
        )
    ) == {"B", "D"}


def test_one_real_fill_is_single_dark_cluster():
    assert cluster(
        {"A": 83, "B": 80, "C": 51, "D": 76},
        {"A": 0, "B": -6, "C": 32, "D": 2},
    ) == ["C"]


def test_blank_row_does_not_create_fill():
    assert cluster(
        {"A": 97, "B": 94, "C": 97, "D": 94},
        {"A": -5, "B": -8, "C": -6, "D": -4},
    ) == []


def test_blurred_printed_glyph_does_not_create_fill():
    assert cluster(
        {"A": 98, "B": 89, "C": 97, "D": 84},
        {"A": -6, "B": -9, "C": -4, "D": -1},
    ) == []


def test_dark_whole_row_does_not_create_fill():
    assert cluster(
        {"A": 76, "B": 77, "C": 74, "D": 70},
        {"A": 6, "B": 13, "C": 15, "D": 23},
    ) == []


def test_column_guard_rejects_option_glyph_split():
    assert cluster(
        {"A": 92, "B": 91, "C": 90, "D": 76},
        {"A": 0, "B": 0, "C": 0, "D": 3},
    ) == []
