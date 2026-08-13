
from pathlib import Path
import csv
import cv2
import numpy as np

# ------------------------------------------------------------
# Local bubble-center diagnostic
# ------------------------------------------------------------
# Purpose:
#   Inspect ONLY Q33, Q34, and Q133 around their current detected
#   centers without changing the scanner.
#
# Input:
#   corrected_omr.jpg
#   recognition_report.csv
#
# Output:
#   local_center_diagnostic.csv
#   local_center_debug/
#
# This script DOES NOT use the answer key.
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "corrected_omr.jpg"
REPORT_PATH = ROOT / "recognition_report.csv"

OUT_CSV = ROOT / "local_center_diagnostic.csv"
OUT_DIR = ROOT / "local_center_debug"

QUESTIONS = [33, 34, 133]
OPTIONS = ["A", "B", "C", "D"]

# Small local search only. This is intentionally much tighter than
# the failed ±20 px shared-row experiment.
DX_VALUES = [-4, -2, 0, 2, 4]
DY_VALUES = [-4, -2, 0, 2, 4]

PATCH_RADIUS = 11
CORE_RADIUS = 5
MICRO_RADIUS = 3
DARK_THRESHOLD = 120


def circular_mask(radius):
    yy, xx = np.ogrid[
        -radius: radius + 1,
        -radius: radius + 1,
    ]
    return (
        xx * xx + yy * yy
        <= radius * radius
    )


def safe_patch(gray, x, y, radius):
    x = int(round(x))
    y = int(round(y))
    r = int(radius)

    x0 = x - r
    y0 = y - r
    x1 = x + r + 1
    y1 = y + r + 1

    if (
        x0 < 0
        or y0 < 0
        or x1 > gray.shape[1]
        or y1 > gray.shape[0]
    ):
        return None

    return gray[y0:y1, x0:x1]


def metrics_at(gray, x, y):
    patch = safe_patch(
        gray,
        x,
        y,
        PATCH_RADIUS,
    )

    if patch is None:
        return None

    disk_mask = circular_mask(
        PATCH_RADIUS
    )

    core_mask_full = circular_mask(
        CORE_RADIUS
    )

    micro_mask_full = circular_mask(
        MICRO_RADIUS
    )

    # Embed smaller masks into the full patch coordinate system.
    side = patch.shape[0]
    center = side // 2

    core_mask = np.zeros_like(
        patch,
        dtype=bool,
    )
    micro_mask = np.zeros_like(
        patch,
        dtype=bool,
    )

    cr = CORE_RADIUS
    mr = MICRO_RADIUS

    core_mask[
        center - cr:center + cr + 1,
        center - cr:center + cr + 1,
    ] = core_mask_full

    micro_mask[
        center - mr:center + mr + 1,
        center - mr:center + mr + 1,
    ] = micro_mask_full

    disk_pixels = patch[
        disk_mask
    ]

    core_pixels = patch[
        core_mask
    ]

    micro_pixels = patch[
        micro_mask
    ]

    center_darkness = float(
        255.0
        -
        np.mean(
            core_pixels
        )
    )

    micro_darkness = float(
        255.0
        -
        np.mean(
            micro_pixels
        )
    )

    disk_dark_ratio = float(
        np.mean(
            disk_pixels
            <
            DARK_THRESHOLD
        )
    )

    core_dark_ratio = float(
        np.mean(
            core_pixels
            <
            DARK_THRESHOLD
        )
    )

    # Edge-vs-center contrast:
    # true fills should have meaningful center darkness; empty printed
    # rings can have dark edges with a much lighter center.
    ring_mask = (
        disk_mask
        &
        ~core_mask
    )

    ring_pixels = patch[
        ring_mask
    ]

    ring_darkness = float(
        255.0
        -
        np.mean(
            ring_pixels
        )
    )

    center_minus_ring = (
        center_darkness
        -
        ring_darkness
    )

    # Composite evidence for diagnostics only.
    evidence = (
        0.35 * center_darkness
        +
        0.30 * micro_darkness
        +
        30.0 * core_dark_ratio
        +
        20.0 * disk_dark_ratio
        +
        0.20 * center_minus_ring
    )

    return {
        "center_darkness":
            center_darkness,

        "micro_darkness":
            micro_darkness,

        "core_dark_ratio":
            core_dark_ratio,

        "disk_dark_ratio":
            disk_dark_ratio,

        "ring_darkness":
            ring_darkness,

        "center_minus_ring":
            center_minus_ring,

        "evidence":
            float(
                evidence
            ),
    }


def load_centers():
    rows = {}

    with REPORT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(
            f
        )

        for row in reader:
            q = int(
                row[
                    "question"
                ]
            )

            if q not in QUESTIONS:
                continue

            option = row[
                "option"
            ]

            rows[
                (
                    q,
                    option,
                )
            ] = {
                "x":
                    float(
                        row[
                            "crop_center_x"
                        ]
                    ),

                "y":
                    float(
                        row[
                            "crop_center_y"
                        ]
                    ),

                "final_answer":
                    row.get(
                        "final_answer",
                        "",
                    ),

                "final_status":
                    row.get(
                        "final_status",
                        "",
                    ),
            }

    return rows


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            IMAGE_PATH
        )

    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            REPORT_PATH
        )

    image = cv2.imread(
        str(
            IMAGE_PATH
        )
    )

    if image is None:
        raise RuntimeError(
            "Could not read corrected_omr.jpg"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    centers = load_centers()

    OUT_DIR.mkdir(
        exist_ok=True
    )

    output_rows = []

    for question in QUESTIONS:
        question_image = (
            image.copy()
        )

        best_by_option = {}

        for option in OPTIONS:
            key = (
                question,
                option,
            )

            if key not in centers:
                continue

            base = centers[
                key
            ]

            x0 = float(
                base[
                    "x"
                ]
            )

            y0 = float(
                base[
                    "y"
                ]
            )

            option_candidates = []

            for dy in DY_VALUES:
                for dx in DX_VALUES:
                    x = x0 + dx
                    y = y0 + dy

                    metrics = metrics_at(
                        gray,
                        x,
                        y,
                    )

                    if metrics is None:
                        continue

                    row = {
                        "question":
                            question,

                        "option":
                            option,

                        "base_x":
                            round(
                                x0,
                                2,
                            ),

                        "base_y":
                            round(
                                y0,
                                2,
                            ),

                        "dx":
                            dx,

                        "dy":
                            dy,

                        "sample_x":
                            round(
                                x,
                                2,
                            ),

                        "sample_y":
                            round(
                                y,
                                2,
                            ),

                        **{
                            key2:
                                round(
                                    value,
                                    5,
                                )
                            for key2, value
                            in metrics.items()
                        },
                    }

                    output_rows.append(
                        row
                    )

                    option_candidates.append(
                        row
                    )

            if not option_candidates:
                continue

            option_candidates.sort(
                key=lambda row:
                    row[
                        "evidence"
                    ],
                reverse=True,
            )

            best = (
                option_candidates[0]
            )

            best_by_option[
                option
            ] = best

            # Draw original center.
            cv2.circle(
                question_image,
                (
                    int(
                        round(
                            x0
                        )
                    ),
                    int(
                        round(
                            y0
                        )
                    ),
                ),
                7,
                (
                    0,
                    165,
                    255,
                ),
                1,
            )

            # Draw locally best center.
            bx = int(
                round(
                    best[
                        "sample_x"
                    ]
                )
            )

            by = int(
                round(
                    best[
                        "sample_y"
                    ]
                )
            )

            cv2.circle(
                question_image,
                (
                    bx,
                    by,
                ),
                4,
                (
                    0,
                    255,
                    0,
                ),
                -1,
            )

            cv2.putText(
                question_image,
                option,
                (
                    bx - 6,
                    by - 12,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (
                    0,
                    0,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

        if not best_by_option:
            continue

        ranked_options = sorted(
            best_by_option.items(),
            key=lambda item:
                item[1][
                    "evidence"
                ],
            reverse=True,
        )

        # Crop around the four bubble columns for this question.
        xs = [
            int(
                round(
                    centers[
                        (
                            question,
                            option,
                        )
                    ][
                        "x"
                    ]
                )
            )
            for option in OPTIONS
            if (
                question,
                option,
            ) in centers
        ]

        ys = [
            int(
                round(
                    centers[
                        (
                            question,
                            option,
                        )
                    ][
                        "y"
                    ]
                )
            )
            for option in OPTIONS
            if (
                question,
                option,
            ) in centers
        ]

        x_min = max(
            0,
            min(
                xs
            )
            -
            40,
        )

        x_max = min(
            image.shape[1],
            max(
                xs
            )
            +
            40,
        )

        y_center = int(
            round(
                np.median(
                    ys
                )
            )
        )

        y_min = max(
            0,
            y_center
            -
            35,
        )

        y_max = min(
            image.shape[0],
            y_center
            +
            36,
        )

        crop = question_image[
            y_min:y_max,
            x_min:x_max,
        ]

        summary = "  ".join(
            f"{option}:{best['evidence']:.1f}"
            for option, best
            in ranked_options
        )

        canvas = cv2.copyMakeBorder(
            crop,
            28,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=255,
        )

        cv2.putText(
            canvas,
            f"Q{question}  {summary}",
            (
                5,
                19,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                0,
                0,
                0,
            ),
            1,
            cv2.LINE_AA,
        )

        cv2.imwrite(
            str(
                OUT_DIR
                /
                f"Q{question:03d}_local_search.jpg"
            ),
            canvas,
        )

    fieldnames = [
        "question",
        "option",
        "base_x",
        "base_y",
        "dx",
        "dy",
        "sample_x",
        "sample_y",
        "center_darkness",
        "micro_darkness",
        "core_dark_ratio",
        "disk_dark_ratio",
        "ring_darkness",
        "center_minus_ring",
        "evidence",
    ]

    with OUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in output_rows:
            writer.writerow(
                row
            )

    print(
        f"Created: {OUT_CSV}"
    )
    print(
        f"Created: {OUT_DIR}"
    )
    print(
        "Questions checked: 33, 34, 133"
    )
    print(
        "Search window: dx/dy = -4,-2,0,2,4"
    )


if __name__ == "__main__":
    main()
