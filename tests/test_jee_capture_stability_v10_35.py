from pathlib import Path

import numpy as np

from jee_reader import _cluster_1d


ROOT = Path(__file__).resolve().parents[1]


def test_cluster_is_permutation_invariant():
    values = []

    for center in (
        100.0,
        130.0,
        160.0,
        190.0,
    ):
        values.extend(
            [
                center - 1.5,
                center - 0.5,
                center,
                center + 0.5,
                center + 1.5,
            ]
        )

    expected = _cluster_1d(
        values,
        4,
    )

    rng = np.random.default_rng(
        12345
    )

    for _ in range(20):
        shuffled = list(
            rng.permutation(values)
        )

        assert _cluster_1d(
            shuffled,
            4,
        ) == expected


def test_cluster_tolerates_missing_ring_detections():
    values = (
        [99.0, 100.0, 101.0, 100.5, 99.5, 101.5]
        + [129.0, 130.0, 131.0, 129.5, 130.5]
        + [159.0, 160.0, 161.0, 160.5, 159.5, 161.5]
        + [189.0, 190.0, 191.0, 189.5, 190.5]
    )

    centres = _cluster_1d(
        values,
        4,
    )

    assert centres is not None

    for actual, expected in zip(
        centres,
        (
            100.0,
            130.0,
            160.0,
            190.0,
        ),
    ):
        assert abs(
            actual - expected
        ) <= 1.5


def test_mcq_uses_affine_lattice_smoothing():
    source = (
        ROOT
        / "jee_reader.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "affine_lattice_smoothing_v10_35"
        in source
    )

    assert "raw_x_centres" in source
    assert "raw_y_centres" in source


def test_numerical_digits_use_affine_projection():
    source = (
        ROOT
        / "jee_reader.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "numeric_affine_sampling_v10_35"
        in source
    )

    assert (
        'project_local_x('
        in source
    )

    assert (
        'project_local_y('
        in source
    )
