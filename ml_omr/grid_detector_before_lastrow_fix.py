from __future__ import annotations

import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

ROI_MARGIN_X = 24
ROI_MARGIN_Y = 18

MIN_SIZE = 10
MAX_SIZE = 28
MIN_AREA = 35
MAX_AREA = 700
MIN_CIRCULARITY = 0.18

MAX_INITIAL_MATCH_DISTANCE = 17.0
MIN_MATCHES_FOR_GRID = 28

# Residual correction is limited so a false contour cannot bend the grid badly.
MAX_ROW_RESIDUAL_X = 7.0
MAX_ROW_RESIDUAL_Y = 7.0


# ============================================================
# CANDIDATE DETECTION
# ============================================================

def _prepare_binary(gray_roi):
    if gray_roi.ndim == 3:
        gray_roi = cv2.cvtColor(
            gray_roi,
            cv2.COLOR_BGR2GRAY,
        )

    clahe = cv2.createCLAHE(
        clipLimit=1.8,
        tileGridSize=(6, 6),
    )

    norm = clahe.apply(
        gray_roi
    )

    blur = cv2.GaussianBlur(
        norm,
        (3, 3),
        0,
    )

    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )

    return binary


def _candidate_score(
    contour,
    x,
    y,
    w,
    h,
):
    area = float(
        cv2.contourArea(
            contour
        )
    )

    perimeter = float(
        cv2.arcLength(
            contour,
            True,
        )
    )

    circularity = (
        4.0
        *
        np.pi
        *
        area
        /
        (
            perimeter
            *
            perimeter
            +
            1e-6
        )
    )

    size = (
        w
        +
        h
    ) / 2.0

    # Bubbles are close to ~20-22 px diameter on canonical sheet.
    size_score = max(
        0.0,
        1.0
        -
        abs(
            size
            -
            20.0
        )
        /
        12.0,
    )

    return float(
        circularity
        *
        2.0
        +
        size_score
    )


def _deduplicate_candidates(
    candidates,
    min_distance=5.0,
):
    candidates = sorted(
        candidates,
        key=lambda item:
            item[
                "score"
            ],
        reverse=True,
    )

    kept = []

    for candidate in candidates:
        cx = candidate[
            "x"
        ]
        cy = candidate[
            "y"
        ]

        duplicate = False

        for previous in kept:
            distance = float(
                np.hypot(
                    cx
                    -
                    previous[
                        "x"
                    ],
                    cy
                    -
                    previous[
                        "y"
                    ],
                )
            )

            if distance < min_distance:
                duplicate = True
                break

        if not duplicate:
            kept.append(
                candidate
            )

    return kept


def detect_circle_candidates(
    gray,
    roi,
):
    """
    Detect actual printed bubble-ring / filled-bubble candidates in a response block.

    roi = (x1, y1, x2, y2)
    Returned coordinates are full-image coordinates.
    """

    x1, y1, x2, y2 = [
        int(
            round(
                value
            )
        )
        for value
        in roi
    ]

    h, w = gray.shape[:2]

    x1 = max(
        0,
        x1,
    )
    y1 = max(
        0,
        y1,
    )
    x2 = min(
        w,
        x2,
    )
    y2 = min(
        h,
        y2,
    )

    crop = gray[
        y1:y2,
        x1:x2,
    ]

    binary = _prepare_binary(
        crop
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(
            contour
        )

        if (
            bw < MIN_SIZE
            or bh < MIN_SIZE
            or bw > MAX_SIZE
            or bh > MAX_SIZE
        ):
            continue

        aspect = (
            float(
                bw
            )
            /
            max(
                1.0,
                float(
                    bh
                ),
            )
        )

        if (
            aspect < 0.55
            or aspect > 1.45
        ):
            continue

        area = float(
            cv2.contourArea(
                contour
            )
        )

        if (
            area < MIN_AREA
            or area > MAX_AREA
        ):
            continue

        perimeter = float(
            cv2.arcLength(
                contour,
                True,
            )
        )

        if perimeter <= 0:
            continue

        circularity = (
            4.0
            *
            np.pi
            *
            area
            /
            (
                perimeter
                *
                perimeter
                +
                1e-6
            )
        )

        if circularity < MIN_CIRCULARITY:
            continue

        center_x = (
            x1
            +
            bx
            +
            bw / 2.0
        )

        center_y = (
            y1
            +
            by
            +
            bh / 2.0
        )

        score = _candidate_score(
            contour,
            bx,
            by,
            bw,
            bh,
        )

        candidates.append(
            {
                "x":
                    float(
                        center_x
                    ),

                "y":
                    float(
                        center_y
                    ),

                "w":
                    int(
                        bw
                    ),

                "h":
                    int(
                        bh
                    ),

                "score":
                    float(
                        score
                    ),
            }
        )

    return _deduplicate_candidates(
        candidates
    )


# ============================================================
# GRID FITTING
# ============================================================

def _block_question_range(
    column_index,
    template,
):
    qpc = int(
        template[
            "questions_per_column"
        ]
    )

    start = (
        column_index
        *
        qpc
        +
        1
    )

    end = (
        start
        +
        qpc
    )

    return (
        start,
        end,
    )


def _block_roi(
    coordinates,
    column_index,
    template,
):
    start, end = _block_question_range(
        column_index,
        template,
    )

    xs = []
    ys = []

    for question in range(
        start,
        end,
    ):
        option_map = coordinates[
            question
        ]

        for x, y in option_map.values():
            xs.append(
                float(
                    x
                )
            )
            ys.append(
                float(
                    y
                )
            )

    return (
        min(
            xs
        )
        -
        ROI_MARGIN_X,

        min(
            ys
        )
        -
        ROI_MARGIN_Y,

        max(
            xs
        )
        +
        ROI_MARGIN_X,

        max(
            ys
        )
        +
        ROI_MARGIN_Y,
    )


def _expected_points(
    coordinates,
    column_index,
    template,
):
    start, end = _block_question_range(
        column_index,
        template,
    )

    options = template[
        "options"
    ]

    expected = []

    for question in range(
        start,
        end,
    ):
        row_index = (
            question
            -
            start
        )

        for option_index, option in enumerate(
            options
        ):
            x, y = coordinates[
                question
            ][
                option
            ]

            expected.append(
                {
                    "question":
                        int(
                            question
                        ),

                    "row":
                        int(
                            row_index
                        ),

                    "option":
                        option,

                    "option_index":
                        int(
                            option_index
                        ),

                    "x":
                        float(
                            x
                        ),

                    "y":
                        float(
                            y
                        ),
                }
            )

    return expected


def _initial_correspondences(
    expected,
    candidates,
):
    """
    Greedy nearest-neighbour assignment from expected lattice to detected circles.
    """

    if not candidates:
        return []

    candidate_xy = np.asarray(
        [
            [
                item[
                    "x"
                ],
                item[
                    "y"
                ],
            ]
            for item
            in candidates
        ],
        dtype=np.float32,
    )

    possible = []

    for expected_index, point in enumerate(
        expected
    ):
        diff = (
            candidate_xy
            -
            np.asarray(
                [
                    point[
                        "x"
                    ],
                    point[
                        "y"
                    ],
                ],
                dtype=np.float32,
            )
        )

        distances = np.sqrt(
            np.sum(
                diff
                *
                diff,
                axis=1,
            )
        )

        candidate_index = int(
            np.argmin(
                distances
            )
        )

        distance = float(
            distances[
                candidate_index
            ]
        )

        if distance <= MAX_INITIAL_MATCH_DISTANCE:
            possible.append(
                (
                    distance,
                    expected_index,
                    candidate_index,
                )
            )

    # Ensure one candidate is not assigned to multiple expected bubbles.
    possible.sort(
        key=lambda item:
            item[
                0
            ]
    )

    used_expected = set()
    used_candidates = set()

    correspondences = []

    for (
        distance,
        expected_index,
        candidate_index,
    ) in possible:
        if expected_index in used_expected:
            continue

        if candidate_index in used_candidates:
            continue

        used_expected.add(
            expected_index
        )

        used_candidates.add(
            candidate_index
        )

        point = expected[
            expected_index
        ]

        candidate = candidates[
            candidate_index
        ]

        correspondences.append(
            {
                **point,

                "detected_x":
                    float(
                        candidate[
                            "x"
                        ]
                    ),

                "detected_y":
                    float(
                        candidate[
                            "y"
                        ]
                    ),

                "distance":
                    float(
                        distance
                    ),
            }
        )

    return correspondences


def _fit_affine(
    correspondences,
):
    if len(
        correspondences
    ) < 6:
        return None, None

    src = np.asarray(
        [
            [
                item[
                    "x"
                ],
                item[
                    "y"
                ],
            ]
            for item
            in correspondences
        ],
        dtype=np.float32,
    )

    dst = np.asarray(
        [
            [
                item[
                    "detected_x"
                ],
                item[
                    "detected_y"
                ],
            ]
            for item
            in correspondences
        ],
        dtype=np.float32,
    )

    matrix, inliers = cv2.estimateAffine2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.5,
        maxIters=2500,
        confidence=0.995,
        refineIters=25,
    )

    return (
        matrix,
        inliers,
    )


def _apply_affine_point(
    matrix,
    x,
    y,
):
    if matrix is None:
        return (
            float(
                x
            ),
            float(
                y
            ),
        )

    point = np.asarray(
        [
            x,
            y,
            1.0,
        ],
        dtype=np.float64,
    )

    out = matrix @ point

    return (
        float(
            out[
                0
            ]
        ),
        float(
            out[
                1
            ]
        ),
    )


def _robust_row_residuals(
    correspondences,
    matrix,
    rows_per_column,
):
    """
    After affine fit, measure remaining local residual per question row.
    Missing rows are interpolated.
    """

    row_dx = {
        row: []
        for row
        in range(
            rows_per_column
        )
    }

    row_dy = {
        row: []
        for row
        in range(
            rows_per_column
        )
    }

    for item in correspondences:
        predicted_x, predicted_y = _apply_affine_point(
            matrix,
            item[
                "x"
            ],
            item[
                "y"
            ],
        )

        dx = (
            item[
                "detected_x"
            ]
            -
            predicted_x
        )

        dy = (
            item[
                "detected_y"
            ]
            -
            predicted_y
        )

        if (
            abs(
                dx
            )
            <=
            MAX_ROW_RESIDUAL_X
            and
            abs(
                dy
            )
            <=
            MAX_ROW_RESIDUAL_Y
        ):
            row_dx[
                item[
                    "row"
                ]
            ].append(
                dx
            )

            row_dy[
                item[
                    "row"
                ]
            ].append(
                dy
            )

    known_rows = []
    known_dx = []
    known_dy = []

    for row in range(
        rows_per_column
    ):
        if (
            row_dx[
                row
            ]
            and
            row_dy[
                row
            ]
        ):
            known_rows.append(
                row
            )

            known_dx.append(
                float(
                    np.median(
                        row_dx[
                            row
                        ]
                    )
                )
            )

            known_dy.append(
                float(
                    np.median(
                        row_dy[
                            row
                        ]
                    )
                )
            )

    if not known_rows:
        return {
            row:
                (
                    0.0,
                    0.0,
                )
            for row
            in range(
                rows_per_column
            )
        }

    # Smooth known residuals before interpolation.
    if len(
        known_dx
    ) >= 3:
        smoothed_dx = []

        smoothed_dy = []

        for i in range(
            len(
                known_rows
            )
        ):
            start = max(
                0,
                i - 1,
            )

            end = min(
                len(
                    known_rows
                ),
                i + 2,
            )

            smoothed_dx.append(
                float(
                    np.median(
                        known_dx[
                            start:end
                        ]
                    )
                )
            )

            smoothed_dy.append(
                float(
                    np.median(
                        known_dy[
                            start:end
                        ]
                    )
                )
            )

        known_dx = smoothed_dx
        known_dy = smoothed_dy

    all_rows = np.arange(
        rows_per_column,
        dtype=np.float32,
    )

    interp_dx = np.interp(
        all_rows,
        np.asarray(
            known_rows,
            dtype=np.float32,
        ),
        np.asarray(
            known_dx,
            dtype=np.float32,
        ),
    )

    interp_dy = np.interp(
        all_rows,
        np.asarray(
            known_rows,
            dtype=np.float32,
        ),
        np.asarray(
            known_dy,
            dtype=np.float32,
        ),
    )

    return {
        int(
            row
        ):
            (
                float(
                    np.clip(
                        interp_dx[
                            row
                        ],
                        -MAX_ROW_RESIDUAL_X,
                        MAX_ROW_RESIDUAL_X,
                    )
                ),
                float(
                    np.clip(
                        interp_dy[
                            row
                        ],
                        -MAX_ROW_RESIDUAL_Y,
                        MAX_ROW_RESIDUAL_Y,
                    )
                ),
            )
        for row
        in range(
            rows_per_column
        )
    }


def fit_response_grid(
    gray,
    coordinates,
    template,
):
    """
    Detect actual response-bubble circles and fit the known 4 x 45 lattice.

    IMPORTANT:
    If an expected bubble has a reliable detected candidate ("pin dot"),
    that detected candidate center becomes the FINAL center directly.

    The affine / row-residual model is used ONLY for bubbles whose actual
    circle could not be matched reliably.

    Returns:
        fitted_coordinates, debug_info
    """

    rows_per_column = len(
        template[
            "question_y_positions"
        ]
    )

    fitted = {}
    debug_info = {}

    for column_index in range(
        len(
            template[
                "columns"
            ]
        )
    ):
        roi = _block_roi(
            coordinates,
            column_index,
            template,
        )

        candidates = detect_circle_candidates(
            gray,
            roi,
        )

        expected = _expected_points(
            coordinates,
            column_index,
            template,
        )

        correspondences = _initial_correspondences(
            expected,
            candidates,
        )

        matrix = None
        inliers = None

        if len(
            correspondences
        ) >= MIN_MATCHES_FOR_GRID:
            matrix, inliers = _fit_affine(
                correspondences
            )

        start_question, end_question = (
            _block_question_range(
                column_index,
                template,
            )
        )

        # Direct lookup of real detected pin centers.
        direct_match = {
            (
                int(
                    item[
                        "question"
                    ]
                ),
                str(
                    item[
                        "option"
                    ]
                ),
            ):
                (
                    float(
                        item[
                            "detected_x"
                        ]
                    ),
                    float(
                        item[
                            "detected_y"
                        ]
                    ),
                    float(
                        item[
                            "distance"
                        ]
                    ),
                )
            for item
            in correspondences
        }

        # If affine fitting failed, still use direct matches wherever possible.
        if matrix is None:
            direct_used = 0

            for question in range(
                start_question,
                end_question,
            ):
                fitted[
                    question
                ] = {}

                for option, (
                    x,
                    y,
                ) in coordinates[
                    question
                ].items():
                    key = (
                        int(
                            question
                        ),
                        str(
                            option
                        ),
                    )

                    if key in direct_match:
                        dx, dy, _distance = direct_match[
                            key
                        ]

                        fitted[
                            question
                        ][
                            option
                        ] = (
                            dx,
                            dy,
                        )

                        direct_used += 1

                    else:
                        fitted[
                            question
                        ][
                            option
                        ] = (
                            float(
                                x
                            ),
                            float(
                                y
                            ),
                        )

            debug_info[
                column_index
            ] = {
                "status":
                    "direct_pin_fallback",

                "roi":
                    [
                        round(
                            float(
                                value
                            ),
                            2,
                        )
                        for value
                        in roi
                    ],

                "candidate_count":
                    len(
                        candidates
                    ),

                "match_count":
                    len(
                        correspondences
                    ),

                "direct_pin_count":
                    int(
                        direct_used
                    ),

                "inlier_count":
                    0,

                "affine":
                    None,
            }

            continue

        residuals = _robust_row_residuals(
            correspondences,
            matrix,
            rows_per_column,
        )

        direct_used = 0
        model_used = 0

        for question in range(
            start_question,
            end_question,
        ):
            row = (
                question
                -
                start_question
            )

            dx_residual, dy_residual = residuals[
                row
            ]

            fitted[
                question
            ] = {}

            for option, (
                x,
                y,
            ) in coordinates[
                question
            ].items():
                key = (
                    int(
                        question
                    ),
                    str(
                        option
                    ),
                )

                # ------------------------------------------------
                # BEST CASE:
                # use the real detected printed-circle center.
                # ------------------------------------------------
                if key in direct_match:
                    pin_x, pin_y, distance = direct_match[
                        key
                    ]

                    fitted[
                        question
                    ][
                        option
                    ] = (
                        pin_x,
                        pin_y,
                    )

                    direct_used += 1

                    continue

                # ------------------------------------------------
                # FALLBACK:
                # no trustworthy pin for this bubble.
                # Use fitted lattice + local row residual.
                # ------------------------------------------------
                fx, fy = _apply_affine_point(
                    matrix,
                    float(
                        x
                    ),
                    float(
                        y
                    ),
                )

                fitted[
                    question
                ][
                    option
                ] = (
                    fx
                    +
                    dx_residual,

                    fy
                    +
                    dy_residual,
                )

                model_used += 1

        inlier_count = (
            int(
                inliers.sum()
            )
            if inliers is not None
            else 0
        )

        debug_info[
            column_index
        ] = {
            "status":
                "pin_locked_grid",

            "roi":
                [
                    round(
                        float(
                            value
                        ),
                        2,
                    )
                    for value
                    in roi
                ],

            "candidate_count":
                len(
                    candidates
                ),

            "match_count":
                len(
                    correspondences
                ),

            "direct_pin_count":
                int(
                    direct_used
                ),

            "model_fallback_count":
                int(
                    model_used
                ),

            "inlier_count":
                inlier_count,

            "affine":
                matrix.tolist(),

            "correspondences":
                correspondences,
        }

    return (
        fitted,
        debug_info,
    )


# ============================================================
# DEBUG DRAWING
# ============================================================

def draw_grid_detection_debug(
    image,
    template,
    input_coordinates,
    fitted_coordinates,
    debug_info,
):
    """
    Blue  = input calibrated coordinates
    Green = detected/fitted grid coordinates
    Yellow dots = detected candidate circles
    """

    debug = image.copy()

    # Draw input/fitted coordinates.
    for question, option_map in fitted_coordinates.items():
        before_map = input_coordinates[
            question
        ]

        for option, (
            fx,
            fy,
        ) in option_map.items():
            bx, by = before_map[
                option
            ]

            cv2.circle(
                debug,
                (
                    int(
                        round(
                            bx
                        )
                    ),
                    int(
                        round(
                            by
                        )
                    ),
                ),
                3,
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
                    int(
                        round(
                            fx
                        )
                    ),
                    int(
                        round(
                            fy
                        )
                    ),
                ),
                7,
                (
                    0,
                    255,
                    0,
                ),
                2,
            )

    # Re-detect candidates for visualization.
    gray = (
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
        if image.ndim == 3
        else image
    )

    for column_index in range(
        len(
            template[
                "columns"
            ]
        )
    ):
        roi = _block_roi(
            input_coordinates,
            column_index,
            template,
        )

        candidates = detect_circle_candidates(
            gray,
            roi,
        )

        for candidate in candidates:
            cv2.circle(
                debug,
                (
                    int(
                        round(
                            candidate[
                                "x"
                            ]
                        )
                    ),
                    int(
                        round(
                            candidate[
                                "y"
                            ]
                        )
                    ),
                ),
                2,
                (
                    0,
                    255,
                    255,
                ),
                -1,
            )

    y_text = 25

    for column_index in sorted(
        debug_info
    ):
        info = debug_info[
            column_index
        ]

        text = (
            f"Grid C{column_index + 1} "
            f"{info.get('status')} "
            f"candidates={info.get('candidate_count', 0)} "
            f"matches={info.get('match_count', 0)} "
            f"pins={info.get('direct_pin_count', 0)} "
            f"inliers={info.get('inlier_count', 0)}"
        )

        cv2.putText(
            debug,
            text,
            (
                15,
                y_text,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (
                0,
                0,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        y_text += 24

    return debug
