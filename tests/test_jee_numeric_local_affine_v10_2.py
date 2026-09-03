from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_numeric_calibration_projects_decimal_and_sign_from_digit_grid(
    monkeypatch,
):
    import jee_reader

    expected_x = [
        100.0, 126.0, 152.0, 178.0, 204.0, 230.0, 256.0
    ]

    expected_y = [
        200.0, 228.0, 256.0, 284.0, 312.0,
        340.0, 368.0, 396.0, 424.0, 452.0
    ]

    actual_x = [
        1.01 * value + 5.0
        for value in expected_x
    ]

    actual_y = [
        1.02 * value + 12.0
        for value in expected_y
    ]

    decimal_y = 150.0
    sign_y = 500.0

    decimal_x = [
        126.0, 152.0, 178.0, 204.0, 230.0
    ]

    circles = []

    for x in actual_x:
        for y in actual_y:
            circles.append(
                (float(x), float(y), 9.0)
            )

    translated_decimal_y = (
        1.02 * decimal_y + 12.0
    )

    for x in decimal_x:
        circles.append(
            (
                1.01 * x + 5.0,
                translated_decimal_y,
                9.0,
            )
        )

    translated_sign = (
        1.01 * expected_x[0] + 5.0,
        1.02 * sign_y + 12.0,
        9.0,
    )

    circles.append(translated_sign)

    monkeypatch.setattr(
        jee_reader,
        "_hough_circles",
        lambda *args, **kwargs: circles,
    )

    question = {
        "question": 71,
        "columns": [
            {
                "x": x,
                "y_positions": expected_y,
                "values": [
                    str(value)
                    for value in range(10)
                ],
            }
            for x in expected_x
        ],
        "decimal_points": [
            {
                "x": x,
                "y": decimal_y,
                "after_column": index + 1,
            }
            for index, x in enumerate(decimal_x)
        ],
        "sign": {
            "x": expected_x[0],
            "y": sign_y,
        },
    }

    template = {
        "bubble_radius": 10,
        "jee_grid_hough_margin": 20,
        "jee_grid_max_calibration_delta": 22,
        "jee_numeric_decimal_match_radius": 14,
    }

    dummy = np.full(
        (700, 400),
        255,
        dtype=np.uint8,
    )

    calibrated, debug = (
        jee_reader._calibrate_numerical_question(
            dummy,
            question,
            template,
        )
    )

    assert debug["calibrated"] is True
    assert debug["digit_row_count"] == 10
    assert (
        debug["calibration_version"]
        == "local_grid_affine_v10_2"
    )

    for decimal in calibrated[
        "decimal_points"
    ]:
        expected_tx = (
            1.01
            * float(decimal["reference_x"])
            + 5.0
        )

        expected_ty = (
            1.02
            * float(decimal["reference_y"])
            + 12.0
        )

        assert abs(
            float(decimal["x"])
            - expected_tx
        ) < 1.0

        assert abs(
            float(decimal["y"])
            - expected_ty
        ) < 1.0

    assert abs(
        float(calibrated["sign"]["x"])
        - translated_sign[0]
    ) < 1.0

    assert abs(
        float(calibrated["sign"]["y"])
        - translated_sign[1]
    ) < 1.0


def test_numeric_calibration_source_uses_local_projection():
    source = (
        ROOT
        / "jee_reader.py"
    ).read_text(encoding="utf-8")

    assert "def project_local_x(" in source
    assert "def project_local_y(" in source
    assert "translated_dx = project_local_x(" in source
    assert "translated_dy = project_local_y(" in source
    assert "translated_sign_x = project_local_x(" in source
    assert '"local_grid_affine_v10_2"' in source
