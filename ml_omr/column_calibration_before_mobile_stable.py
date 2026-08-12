from __future__ import annotations

import cv2
import numpy as np


# ============================================================
# LOCAL BUBBLE-CENTER SEARCH
# ============================================================

def _local_search_best_center(
    gray,
    x,
    y,
    search_radius=8,
    patch_radius=10,
):
    """
    Fine local center search around the JSON coordinate.

    This is deliberately local: it corrects small residual perspective /
    print / camera errors after the main canonical warp.
    """

    h, w = gray.shape[:2]

    best_score = -1e9
    best_xy = (
        int(round(x)),
        int(round(y)),
    )

    for dy in range(
        -search_radius,
        search_radius + 1,
        2,
    ):
        for dx in range(
            -search_radius,
            search_radius + 1,
            2,
        ):
            cx = int(
                round(
                    x + dx
                )
            )

            cy = int(
                round(
                    y + dy
                )
            )

            x1 = max(
                0,
                cx - patch_radius,
            )
            y1 = max(
                0,
                cy - patch_radius,
            )
            x2 = min(
                w,
                cx + patch_radius + 1,
            )
            y2 = min(
                h,
                cy + patch_radius + 1,
            )

            patch = gray[
                y1:y2,
                x1:x2,
            ]

            if (
                patch.shape[0] < 15
                or patch.shape[1] < 15
            ):
                continue

            ph, pw = patch.shape[:2]

            yy, xx = np.ogrid[
                :ph,
                :pw,
            ]

            pcx = (
                pw - 1
            ) / 2.0

            pcy = (
                ph - 1
            ) / 2.0

            rr = np.sqrt(
                (
                    xx - pcx
                ) ** 2
                +
                (
                    yy - pcy
                ) ** 2
            )

            ring_mask = (
                (
                    rr
                    >= patch_radius * 0.55
                )
                &
                (
                    rr
                    <= patch_radius * 0.95
                )
            )

            center_mask = (
                rr
                <= patch_radius * 0.30
            )

            if (
                not np.any(
                    ring_mask
                )
                or
                not np.any(
                    center_mask
                )
            ):
                continue

            ring_mean = float(
                np.mean(
                    patch[
                        ring_mask
                    ]
                )
            )

            center_mean = float(
                np.mean(
                    patch[
                        center_mask
                    ]
                )
            )

            # Prefer a clear circular printed ring.
            # A filled center is allowed; center_mean has only a small weight.
            score = (
                (
                    255.0
                    -
                    ring_mean
                )
                +
                0.10
                *
                center_mean
            )

            if score > best_score:
                best_score = score
                best_xy = (
                    cx,
                    cy,
                )

    return (
        best_xy,
        float(
            best_score
        ),
    )


# ============================================================
# ROBUST HELPERS
# ============================================================

def _robust_median(
    values,
):
    if not values:
        return 0.0

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    median = float(
        np.median(
            values
        )
    )

    deviation = np.abs(
        values - median
    )

    mad = float(
        np.median(
            deviation
        )
    )

    if mad <= 1e-6:
        return median

    keep = (
        deviation
        <= max(
            2.0,
            3.0 * mad,
        )
    )

    filtered = values[
        keep
    ]

    if filtered.size == 0:
        return median

    return float(
        np.median(
            filtered
        )
    )


def _interpolate_offset(
    row_index,
    anchors,
):
    """
    Piecewise-linear interpolation between top/middle/bottom calibration
    anchors for one response column.
    """

    if not anchors:
        return (
            0.0,
            0.0,
        )

    anchors = sorted(
        anchors,
        key=lambda item:
            item[
                "row"
            ],
    )

    if row_index <= anchors[0]["row"]:
        return (
            float(
                anchors[0][
                    "dx"
                ]
            ),
            float(
                anchors[0][
                    "dy"
                ]
            ),
        )

    if row_index >= anchors[-1]["row"]:
        return (
            float(
                anchors[-1][
                    "dx"
                ]
            ),
            float(
                anchors[-1][
                    "dy"
                ]
            ),
        )

    for left, right in zip(
        anchors[:-1],
        anchors[1:],
    ):
        if (
            left["row"]
            <= row_index
            <= right["row"]
        ):
            span = max(
                1,
                right["row"]
                -
                left["row"],
            )

            t = (
                row_index
                -
                left["row"]
            ) / float(
                span
            )

            dx = (
                float(
                    left[
                        "dx"
                    ]
                )
                +
                t
                *
                (
                    float(
                        right[
                            "dx"
                        ]
                    )
                    -
                    float(
                        left[
                            "dx"
                        ]
                    )
                )
            )

            dy = (
                float(
                    left[
                        "dy"
                    ]
                )
                +
                t
                *
                (
                    float(
                        right[
                            "dy"
                        ]
                    )
                    -
                    float(
                        left[
                            "dy"
                        ]
                    )
                )
            )

            return (
                dx,
                dy,
            )

    return (
        0.0,
        0.0,
    )


# ============================================================
# AUTO CALIBRATION
# ============================================================

def auto_calibrate_neet_columns(
    gray,
    template,
    search_radius=8,
):
    """
    Auto-correct slight unevenness instead of rejecting it.

    Each of the 4 response columns gets THREE local calibration anchors:
      - top
      - middle
      - bottom

    Runtime coordinates are then interpolated between them, so a scan may
    have slightly different dx/dy near the top vs bottom.

    The template JSON remains unchanged.
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

    row_count = len(
        y_positions
    )

    # Three local bands, with enough samples to suppress filled-bubble outliers.
    band_centers = [
        int(
            round(
                row_count * 0.15
            )
        ),
        int(
            round(
                row_count * 0.50
            )
        ),
        int(
            round(
                row_count * 0.84
            )
        ),
    ]

    band_half_width = 4

    calibration = {}

    for column_index, column in enumerate(
        columns
    ):
        anchors = []

        for center_row in band_centers:

            dx_values = []
            dy_values = []

            start_row = max(
                0,
                center_row
                -
                band_half_width,
            )

            end_row = min(
                row_count,
                center_row
                +
                band_half_width
                +
                1,
            )

            for row_index in range(
                start_row,
                end_row,
            ):
                y = y_positions[
                    row_index
                ]

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
                        search_radius=
                            search_radius,
                        patch_radius=10,
                    )

                    dx_values.append(
                        detected_xy[0]
                        -
                        x
                    )

                    dy_values.append(
                        detected_xy[1]
                        -
                        y
                    )

            dx = _robust_median(
                dx_values
            )

            dy = _robust_median(
                dy_values
            )

            # Fine correction only, but allow realistic handheld residuals.
            dx = float(
                np.clip(
                    dx,
                    -12.0,
                    12.0,
                )
            )

            dy = float(
                np.clip(
                    dy,
                    -12.0,
                    12.0,
                )
            )

            anchors.append(
                {
                    "row":
                        int(
                            center_row
                        ),

                    "dx":
                        round(
                            dx,
                            2,
                        ),

                    "dy":
                        round(
                            dy,
                            2,
                        ),
                }
            )

        calibration[
            column_index
        ] = {
            "anchors":
                anchors,

            # Summary values retained for compatibility/debug.
            "dx":
                round(
                    float(
                        np.median(
                            [
                                anchor[
                                    "dx"
                                ]
                                for anchor
                                in anchors
                            ]
                        )
                    ),
                    2,
                ),

            "dy":
                round(
                    float(
                        np.median(
                            [
                                anchor[
                                    "dy"
                                ]
                                for anchor
                                in anchors
                            ]
                        )
                    ),
                    2,
                ),
        }

    return calibration


# ============================================================
# VALIDATION
# ============================================================

def validate_column_alignment(
    column_offsets,
    hard_limit=12.0,
    max_local_jump=9.0,
):
    """
    Do NOT reject normal slight unevenness.

    We only reject:
      - corrections hitting an extreme hard limit
      - one local band jumping implausibly far from the next

    Normal differences between columns and top/middle/bottom are expected
    and are corrected automatically.
    """

    for column_index, data in (
        column_offsets.items()
    ):

        anchors = data.get(
            "anchors",
            [],
        )

        if not anchors:
            continue

        for anchor in anchors:
            dx = abs(
                float(
                    anchor.get(
                        "dx",
                        0.0,
                    )
                )
            )

            dy = abs(
                float(
                    anchor.get(
                        "dy",
                        0.0,
                    )
                )
            )

            if (
                dx >= hard_limit
                or dy >= hard_limit
            ):
                raise ValueError(
                    "OMR image is too distorted for reliable reading. "
                    "Keep the full sheet visible and scan again."
                )

        for first, second in zip(
            anchors[:-1],
            anchors[1:],
        ):
            dx_jump = abs(
                float(
                    second[
                        "dx"
                    ]
                )
                -
                float(
                    first[
                        "dx"
                    ]
                )
            )

            dy_jump = abs(
                float(
                    second[
                        "dy"
                    ]
                )
                -
                float(
                    first[
                        "dy"
                    ]
                )
            )

            if (
                dx_jump > max_local_jump
                or dy_jump > max_local_jump
            ):
                raise ValueError(
                    "OMR has severe local distortion. "
                    "Please flatten the paper and scan again."
                )

    return {
        "status":
            "auto_corrected",
    }


# ============================================================
# COORDINATE GENERATION
# ============================================================

def generate_calibrated_bubble_coordinates(
    template,
    column_offsets=None,
):
    """
    Build runtime coordinates with per-column, top-to-bottom interpolation.

    This is the key change that lets slight uneven perspective reconfigure
    itself automatically.
    """

    columns = template[
        "columns"
    ]

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

    if column_offsets is None:
        column_offsets = {
            i: {
                "anchors": [
                    {
                        "row":
                            0,

                        "dx":
                            0.0,

                        "dy":
                            0.0,
                    },

                    {
                        "row":
                            len(
                                y_positions
                            )
                            -
                            1,

                        "dx":
                            0.0,

                        "dy":
                            0.0,
                    },
                ],

                "dx":
                    0.0,

                "dy":
                    0.0,
            }
            for i in range(
                len(
                    columns
                )
            )
        }

    coordinates = {}

    for column_index, column in enumerate(
        columns
    ):

        data = column_offsets.get(
            column_index,
            {},
        )

        anchors = data.get(
            "anchors"
        )

        if not anchors:
            dx = float(
                data.get(
                    "dx",
                    0.0,
                )
            )

            dy = float(
                data.get(
                    "dy",
                    0.0,
                )
            )

            anchors = [
                {
                    "row":
                        0,

                    "dx":
                        dx,

                    "dy":
                        dy,
                },

                {
                    "row":
                        len(
                            y_positions
                        )
                        -
                        1,

                    "dx":
                        dx,

                    "dy":
                        dy,
                },
            ]

        for row_index, y in enumerate(
            y_positions
        ):

            dx, dy = _interpolate_offset(
                row_index,
                anchors,
            )

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


# ============================================================
# DEBUG
# ============================================================

def draw_calibration_debug(
    corrected_image,
    template,
    column_offsets,
):
    """
    Green circles = actual runtime calibrated coordinates.

    Red text shows top/mid/bottom anchors for each response column.
    """

    debug = corrected_image.copy()

    coordinates = (
        generate_calibrated_bubble_coordinates(
            template,
            column_offsets,
        )
    )

    for question, option_map in (
        coordinates.items()
    ):
        for option, (
            x,
            y,
        ) in option_map.items():
            cv2.circle(
                debug,
                (
                    int(x),
                    int(y),
                ),
                8,
                (
                    0,
                    255,
                    0,
                ),
                2,
            )

    text_y = 28

    for column_index in sorted(
        column_offsets
    ):

        anchors = (
            column_offsets[
                column_index
            ].get(
                "anchors",
                [],
            )
        )

        anchor_text = " | ".join(
            (
                f"r{a['row']}:"
                f"dx={a['dx']},"
                f"dy={a['dy']}"
            )
            for a in anchors
        )

        text = (
            f"Column {column_index + 1} "
            f"{anchor_text}"
        )

        cv2.putText(
            debug,
            text,
            (
                18,
                text_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                0,
                0,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        text_y += 28

    return debug
