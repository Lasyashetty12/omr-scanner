from __future__ import annotations

import cv2
import numpy as np


def _local_search_best_center(
    gray,
    x,
    y,
    search_radius=6,
    patch_radius=10,
):
    """
    Find the local center that best matches a printed OMR bubble.

    We search a small neighborhood around the JSON coordinate and score
    candidate centers using circular dark-ring structure. This is intended
    for fine correction only, not large realignment.
    """
    h, w = gray.shape[:2]

    best_score = -1e9
    best_xy = (int(round(x)), int(round(y)))

    for dy in range(-search_radius, search_radius + 1, 2):
        for dx in range(-search_radius, search_radius + 1, 2):
            cx = int(round(x + dx))
            cy = int(round(y + dy))

            x1 = max(0, cx - patch_radius)
            y1 = max(0, cy - patch_radius)
            x2 = min(w, cx + patch_radius + 1)
            y2 = min(h, cy + patch_radius + 1)

            patch = gray[y1:y2, x1:x2]

            if patch.shape[0] < patch_radius * 2 - 2:
                continue

            if patch.shape[1] < patch_radius * 2 - 2:
                continue

            ph, pw = patch.shape[:2]
            yy, xx = np.ogrid[:ph, :pw]
            pcx = (pw - 1) / 2.0
            pcy = (ph - 1) / 2.0

            rr = np.sqrt(
                (xx - pcx) ** 2
                +
                (yy - pcy) ** 2
            )

            # Printed bubble ring is expected roughly here.
            ring_mask = (
                (rr >= patch_radius * 0.55)
                &
                (rr <= patch_radius * 0.95)
            )

            center_mask = (
                rr <= patch_radius * 0.35
            )

            if not np.any(ring_mask):
                continue

            ring_mean = float(
                np.mean(
                    patch[ring_mask]
                )
            )

            center_mean = float(
                np.mean(
                    patch[center_mask]
                )
            )

            # Prefer a darker printed ring while avoiding obviously wrong
            # dark blobs outside the expected bubble structure.
            score = (
                (255.0 - ring_mean)
                +
                0.20 * center_mean
            )

            if score > best_score:
                best_score = score
                best_xy = (cx, cy)

    return best_xy, float(best_score)


def auto_calibrate_neet_columns(
    gray,
    template,
    sample_every=5,
    search_radius=6,
):
    """
    Estimate small per-column dx/dy offsets while preserving the original
    JSON template unchanged.

    We sample printed bubble centers across each of the four NEET columns,
    detect their local best centers, then compute a robust median offset.

    Returns:
        {
            0: {"dx": ..., "dy": ...},
            1: {"dx": ..., "dy": ...},
            2: {"dx": ..., "dy": ...},
            3: {"dx": ..., "dy": ...},
        }
    """

    if gray.ndim == 3:
        gray = cv2.cvtColor(
            gray,
            cv2.COLOR_BGR2GRAY,
        )

    columns = template[
        "columns"
    ]

    y_positions = template[
        "question_y_positions"
    ]

    options = template[
        "options"
    ]

    offsets = {}

    for column_index, column in enumerate(columns):

        dx_values = []
        dy_values = []

        for row_index in range(
            0,
            len(y_positions),
            sample_every,
        ):
            y = y_positions[
                row_index
            ]

            # Use all option bubbles, but robust median will suppress
            # filled/dirty outliers.
            for option in options:
                x = column[
                    option
                ]

                (
                    detected_xy,
                    _
                ) = _local_search_best_center(
                    gray,
                    x,
                    y,
                    search_radius=search_radius,
                    patch_radius=10,
                )

                dx_values.append(
                    detected_xy[0] - x
                )

                dy_values.append(
                    detected_xy[1] - y
                )

        if dx_values:
            dx = float(
                np.median(
                    dx_values
                )
            )
        else:
            dx = 0.0

        if dy_values:
            dy = float(
                np.median(
                    dy_values
                )
            )
        else:
            dy = 0.0

        # Fine correction only. Never allow this stage to become a large warp.
        dx = float(
            np.clip(
                dx,
                -8.0,
                8.0,
            )
        )

        dy = float(
            np.clip(
                dy,
                -8.0,
                8.0,
            )
        )

        offsets[
            column_index
        ] = {
            "dx": round(
                dx,
                2,
            ),
            "dy": round(
                dy,
                2,
            ),
        }

    return offsets


def generate_calibrated_bubble_coordinates(
    template,
    column_offsets=None,
):
    """
    Build runtime bubble coordinates while leaving template JSON untouched.
    """

    if column_offsets is None:
        column_offsets = {
            i: {
                "dx": 0.0,
                "dy": 0.0,
            }
            for i in range(
                len(
                    template[
                        "columns"
                    ]
                )
            )
        }

    coordinates = {}

    questions_per_column = int(
        template[
            "questions_per_column"
        ]
    )

    options = template[
        "options"
    ]

    y_positions = template[
        "question_y_positions"
    ]

    for column_index, column in enumerate(
        template[
            "columns"
        ]
    ):
        offset = column_offsets.get(
            column_index,
            {
                "dx": 0.0,
                "dy": 0.0,
            },
        )

        dx = float(
            offset.get(
                "dx",
                0.0,
            )
        )

        dy = float(
            offset.get(
                "dy",
                0.0,
            )
        )

        for row_index, y in enumerate(
            y_positions
        ):
            question = (
                column_index
                *
                questions_per_column
                +
                row_index
                +
                1
            )

            coordinates[
                question
            ] = {}

            for option in options:
                coordinates[
                    question
                ][
                    option
                ] = (
                    int(
                        round(
                            column[
                                option
                            ]
                            +
                            dx
                        )
                    ),
                    int(
                        round(
                            y
                            +
                            dy
                        )
                    ),
                )

    return coordinates


def draw_calibration_debug(
    corrected_image,
    template,
    column_offsets,
):
    """
    Visualize original JSON coordinates vs calibrated runtime coordinates.

    Blue  = original JSON center
    Green = calibrated center
    """

    debug = corrected_image.copy()

    original = generate_calibrated_bubble_coordinates(
        template,
        {
            i: {
                "dx": 0.0,
                "dy": 0.0,
            }
            for i in range(
                len(
                    template[
                        "columns"
                    ]
                )
            )
        },
    )

    calibrated = generate_calibrated_bubble_coordinates(
        template,
        column_offsets,
    )

    for question in calibrated:
        for option in template[
            "options"
        ]:
            ox, oy = original[
                question
            ][
                option
            ]

            cx, cy = calibrated[
                question
            ][
                option
            ]

            cv2.circle(
                debug,
                (
                    int(ox),
                    int(oy),
                ),
                5,
                (
                    255,
                    0,
                    0,
                ),
                1,
            )

            cv2.circle(
                debug,
                (
                    int(cx),
                    int(cy),
                ),
                8,
                (
                    0,
                    255,
                    0,
                ),
                2,
            )

    y = 35

    for column_index in sorted(
        column_offsets
    ):
        offset = column_offsets[
            column_index
        ]

        text = (
            f"Column {column_index + 1}: "
            f"dx={offset['dx']} "
            f"dy={offset['dy']}"
        )

        cv2.putText(
            debug,
            text,
            (
                25,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                0,
                0,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        y += 35

    return debug
