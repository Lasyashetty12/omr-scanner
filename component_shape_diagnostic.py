
from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "corrected_omr.jpg"
REPORT_PATH = ROOT / "recognition_report.csv"

OUT_CSV = ROOT / "component_shape_diagnostic.csv"
OUT_DIR = ROOT / "component_shape_debug"

QUESTIONS = [33, 34]
OPTIONS = ["A", "B", "C", "D"]

PATCH_RADIUS = 16
CENTER_RADIUS = 11
THRESHOLDS = [80, 100, 120, 140, 160]


def crop(gray, x, y, radius):
    x = int(round(x))
    y = int(round(y))
    r = int(radius)
    x0, y0 = x-r, y-r
    x1, y1 = x+r+1, y+r+1
    if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0]:
        return None
    return gray[y0:y1, x0:x1]


def circle_mask(size, radius):
    h = w = size
    cy = h // 2
    cx = w // 2
    yy, xx = np.ogrid[:h, :w]
    return ((xx-cx)**2 + (yy-cy)**2) <= radius*radius


def largest_component_stats(binary, center_xy):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )
    if num <= 1:
        return {
            "area": 0,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "centroid_distance": np.nan,
            "bbox_w": 0,
            "bbox_h": 0,
            "solidity_proxy": 0.0,
        }

    best_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = stats[best_idx, cv2.CC_STAT_LEFT]
    y = stats[best_idx, cv2.CC_STAT_TOP]
    w = stats[best_idx, cv2.CC_STAT_WIDTH]
    h = stats[best_idx, cv2.CC_STAT_HEIGHT]
    area = stats[best_idx, cv2.CC_STAT_AREA]
    cx, cy = centroids[best_idx]
    center_x, center_y = center_xy
    dist = float(np.hypot(cx-center_x, cy-center_y))
    solidity_proxy = float(area / max(1, w*h))

    return {
        "area": int(area),
        "centroid_x": float(cx),
        "centroid_y": float(cy),
        "centroid_distance": dist,
        "bbox_w": int(w),
        "bbox_h": int(h),
        "solidity_proxy": solidity_proxy,
    }


def radial_profile(gray_patch):
    h, w = gray_patch.shape
    cy = h // 2
    cx = w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((xx-cx)**2 + (yy-cy)**2)

    bands = [(0,3), (3,6), (6,9), (9,12), (12,15)]
    out = {}
    for i, (r0, r1) in enumerate(bands):
        mask = (rr >= r0) & (rr < r1)
        vals = gray_patch[mask]
        out[f"radial_dark_{i}"] = float(255.0 - np.mean(vals)) if vals.size else 0.0
    return out


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(IMAGE_PATH)
    if not REPORT_PATH.exists():
        raise FileNotFoundError(REPORT_PATH)

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError("Could not read corrected_omr.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    centers = {}
    with REPORT_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            q = int(row["question"])
            if q not in QUESTIONS:
                continue
            centers[(q, row["option"])] = (
                float(row["crop_center_x"]),
                float(row["crop_center_y"]),
            )

    OUT_DIR.mkdir(exist_ok=True)
    rows = []

    for q in QUESTIONS:
        dbg = image.copy()

        for option in OPTIONS:
            if (q, option) not in centers:
                continue

            x, y = centers[(q, option)]
            patch = crop(gray, x, y, PATCH_RADIUS)
            if patch is None:
                continue

            size = patch.shape[0]
            center = size // 2
            disk_mask = circle_mask(size, CENTER_RADIUS)

            base = {
                "question": q,
                "option": option,
                "center_x": x,
                "center_y": y,
            }
            base.update(radial_profile(patch))

            for thr in THRESHOLDS:
                # dark pixel mask, restricted to bubble disk
                dark = (patch < thr) & disk_mask

                stats = largest_component_stats(
                    dark,
                    (center, center),
                )

                row = dict(base)
                row.update({
                    "threshold": thr,
                    "dark_pixel_count": int(np.sum(dark)),
                    "dark_fraction": float(np.mean(dark[disk_mask])),
                    **stats,
                })
                rows.append(row)

            # Draw center and crop box
            cv2.circle(
                dbg,
                (int(round(x)), int(round(y))),
                CENTER_RADIUS,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                dbg,
                f"Q{q}{option}",
                (int(round(x))-16, int(round(y))-18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        xs = [int(round(centers[(q,o)][0])) for o in OPTIONS if (q,o) in centers]
        ys = [int(round(centers[(q,o)][1])) for o in OPTIONS if (q,o) in centers]
        x0 = max(0, min(xs)-50)
        x1 = min(image.shape[1], max(xs)+50)
        yc = int(round(np.median(ys)))
        y0 = max(0, yc-45)
        y1 = min(image.shape[0], yc+46)

        cv2.imwrite(
            str(OUT_DIR / f"Q{q:03d}_component_shape.jpg"),
            dbg[y0:y1, x0:x1],
        )

    fieldnames = [
        "question","option","center_x","center_y","threshold",
        "dark_pixel_count","dark_fraction","area",
        "centroid_x","centroid_y","centroid_distance",
        "bbox_w","bbox_h","solidity_proxy",
        "radial_dark_0","radial_dark_1","radial_dark_2",
        "radial_dark_3","radial_dark_4",
    ]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created: {OUT_CSV}")
    print(f"Created: {OUT_DIR}")
    print("Questions checked: 33 and 34")
    print("No scanner code was changed.")


if __name__ == "__main__":
    main()
