
import json
from pathlib import Path

import cv2
import numpy as np

import identity_reader


ROOT = Path(__file__).resolve().parents[1]


def _load_template(name):
    return json.loads(
        (
            ROOT
            / "templates"
            / f"{name}.json"
        ).read_text(
            encoding="utf-8"
        )
    )


def _make_identity_row(
    template,
    config,
    filled_label=None,
    shift=(0, 0),
    background=184,
):
    height = int(
        template[
            "sheet_height"
        ]
    )

    width = int(
        template[
            "sheet_width"
        ]
    )

    gray = np.full(
        (
            height,
            width,
        ),
        background,
        dtype=np.uint8,
    )

    # Mild left-to-right lighting gradient.
    gradient = np.linspace(
        -18,
        14,
        width,
        dtype=np.float32,
    )

    gray = np.clip(
        gray.astype(
            np.float32
        )
        + gradient[
            None,
            :
        ],
        0,
        255,
    ).astype(
        np.uint8
    )

    dx, dy = shift

    for label, point in (
        config[
            "choices"
        ].items()
    ):
        x = int(
            round(
                point[0]
                + dx
            )
        )

        y = int(
            round(
                point[1]
                + dy
            )
        )

        # Printed empty bubble ring.
        cv2.circle(
            gray,
            (
                x,
                y,
            ),
            int(
                template.get(
                    "bubble_radius",
                    11,
                )
            ),
            65,
            2,
            lineType=cv2.LINE_AA,
        )

        if (
            filled_label
            is not None
            and str(
                label
            )
            == str(
                filled_label
            )
        ):
            cv2.circle(
                gray,
                (
                    x,
                    y,
                ),
                8,
                48,
                -1,
                lineType=cv2.LINE_AA,
            )

    return gray


def test_kcet_exam_bubble_uses_json_and_cv():
    template = _load_template(
        "kcet"
    )

    config = template[
        "identity"
    ][
        "exam"
    ]

    gray = _make_identity_row(
        template,
        config,
        filled_label="KCET",
        shift=(
            6,
            -5,
        ),
        background=172,
    )

    result = (
        identity_reader._detect_choice_row(
            gray,
            config,
            template,
        )
    )

    assert result[
        "value"
    ] == "KCET"

    assert result[
        "reader"
    ] == "cv_json_choice_v10_16"

    assert result[
        "decision_method"
    ] == "cv_json_solid_fill"

    assert (
        result[
            "scores"
        ][
            "KCET"
        ]
        > result[
            "scores"
        ][
            "NEET"
        ]
    )


def test_neet_exam_bubble_uses_same_json_cv_reader():
    template = _load_template(
        "neet"
    )

    config = template[
        "identity"
    ][
        "exam"
    ]

    gray = _make_identity_row(
        template,
        config,
        filled_label="NEET",
        shift=(
            -5,
            4,
        ),
        background=190,
    )

    result = (
        identity_reader._detect_choice_row(
            gray,
            config,
            template,
        )
    )

    assert result[
        "value"
    ] == "NEET"


def test_class_ii_bubble_uses_same_reader():
    template = _load_template(
        "kcet"
    )

    config = template[
        "identity"
    ][
        "class"
    ]

    gray = _make_identity_row(
        template,
        config,
        filled_label="II",
        shift=(
            4,
            -3,
        ),
        background=176,
    )

    result = (
        identity_reader._detect_choice_row(
            gray,
            config,
            template,
        )
    )

    assert result[
        "value"
    ] == "II"


def test_empty_exam_row_stays_unselected():
    template = _load_template(
        "kcet"
    )

    config = template[
        "identity"
    ][
        "exam"
    ]

    gray = _make_identity_row(
        template,
        config,
        filled_label=None,
        background=182,
    )

    result = (
        identity_reader._detect_choice_row(
            gray,
            config,
            template,
        )
    )

    assert result[
        "value"
    ] is None


def test_identity_json_contains_cv_choice_tuning():
    for name in (
        "kcet",
        "neet",
    ):
        template = _load_template(
            name
        )

        for field in (
            "class",
            "exam",
        ):
            config = template[
                "identity"
            ][
                field
            ]

            assert (
                config[
                    "cv_search_radius"
                ]
                >= 1
            )

            assert (
                config[
                    "cv_filled_threshold"
                ]
                > 0
            )

            assert (
                config[
                    "cv_minimum_confidence_gap"
                ]
                > 0
            )

            assert (
                config[
                    "cv_solid_kernel"
                ]
                >= 1
            )


def test_jee_roll_reader_is_untouched():
    source = (
        ROOT
        / "identity_reader.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "jee_roll_ml_disk_v10_14"
        in source
    )
