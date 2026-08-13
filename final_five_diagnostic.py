
from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "corrected_omr.jpg"
REPORT_PATH = ROOT / "recognition_report.csv"

OUT_CSV = ROOT / "final_five_diagnostic.csv"
OUT_DIR = ROOT / "final_five_debug"

QUESTIONS = [33, 34, 53, 103, 143]
OPTIONS = ["A", "B", "C", "D"]

def crop(gray, x, y, r):
    x = int(round(float(x)))
    y = int(round(float(y)))
    if x-r < 0 or y-r < 0 or x+r+1 > gray.shape[1] or y+r+1 > gray.shape[0]:
        return None
    return gray[y-r:y+r+1, x-r:x+r+1]

def circle_metrics(gray, x, y):
    patch = crop(gray, x, y, 14)
    if patch is None:
        return None

    h, w = patch.shape
    cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((xx-cx)**2 + (yy-cy)**2)

    inner = rr <= 5
    disk = rr <= 11
    ring = (rr >= 8) & (rr <= 12)

    return {
        "inner_darkness": float(255 - np.mean(patch[inner])),
        "disk_darkness": float(255 - np.mean(patch[disk])),
        "ring_darkness": float(255 - np.mean(patch[ring])),
        "dark_frac_100": float(np.mean(patch[disk] < 100)),
        "dark_frac_120": float(np.mean(patch[disk] < 120)),
        "dark_frac_140": float(np.mean(patch[disk] < 140)),
    }

def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(IMAGE_PATH)
    if not REPORT_PATH.exists():
        raise FileNotFoundError(REPORT_PATH)

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError("Could not read corrected_omr.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    report = {}
    with REPORT_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            q = int(row["question"])
            if q not in QUESTIONS and q not in (44,45,89,90,134,135,179,180):
                continue
            report[(q, row["option"])] = row

    OUT_DIR.mkdir(exist_ok=True)
    out_rows = []

    # Five-question local diagnostic, only +/-3 px so it cannot jump rows.
    for q in QUESTIONS:
        dbg = image.copy()
        xs, ys = [], []

        for option in OPTIONS:
            row = report.get((q, option))
            if not row:
                continue

            x0 = float(row["crop_center_x"])
            y0 = float(row["crop_center_y"])
            xs.append(int(round(x0)))
            ys.append(int(round(y0)))

            best = None
            for dy in (-3,-2,-1,0,1,2,3):
                for dx in (-3,-2,-1,0,1,2,3):
                    m = circle_metrics(gray, x0+dx, y0+dy)
                    if m is None:
                        continue
                    score = (
                        0.50*m["inner_darkness"]
                        + 0.25*m["disk_darkness"]
                        + 40*m["dark_frac_120"]
                        - 0.08*(abs(dx)+abs(dy))
                    )
                    candidate = (score, dx, dy, m)
                    if best is None or candidate[0] > best[0]:
                        best = candidate

            if best:
                score, dx, dy, m = best
                out_rows.append({
                    "question": q,
                    "option": option,
                    "base_x": x0,
                    "base_y": y0,
                    "best_dx": dx,
                    "best_dy": dy,
                    "score": score,
                    **m,
                    "final_answer": row.get("final_answer",""),
                    "final_status": row.get("final_status",""),
                })

                bx = int(round(x0+dx))
                by = int(round(y0+dy))
                cv2.circle(dbg, (int(round(x0)), int(round(y0))), 8, (0,165,255), 1)
                cv2.circle(dbg, (bx,by), 3, (0,255,0), -1)
                cv2.putText(dbg, option, (bx-5,by-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1, cv2.LINE_AA)

        if xs and ys:
            x0 = max(0, min(xs)-45)
            x1 = min(image.shape[1], max(xs)+45)
            yc = int(round(np.median(ys)))
            y0 = max(0, yc-40)
            y1 = min(image.shape[0], yc+41)
            cv2.imwrite(str(OUT_DIR / f"Q{q:03d}_diagnostic.jpg"), dbg[y0:y1, x0:x1])

    # Final-row physical-circle check.
    # Compares q44->q45, q89->q90, q134->q135, q179->q180.
    for q_prev, q_last in ((44,45),(89,90),(134,135),(179,180)):
        for option in OPTIONS:
            prev = report.get((q_prev, option))
            last = report.get((q_last, option))
            if not prev or not last:
                continue

            for label, row in (("previous",prev),("last",last)):
                x = float(row["crop_center_x"])
                y = float(row["crop_center_y"])
                m = circle_metrics(gray, x, y)
                if m:
                    out_rows.append({
                        "question": q_last,
                        "option": option,
                        "base_x": x,
                        "base_y": y,
                        "best_dx": 0,
                        "best_dy": 0,
                        "score": "",
                        **m,
                        "final_answer": f"FINAL_ROW_{label.upper()}",
                        "final_status": "",
                    })

    fieldnames = [
        "question","option","base_x","base_y","best_dx","best_dy","score",
        "inner_darkness","disk_darkness","ring_darkness",
        "dark_frac_100","dark_frac_120","dark_frac_140",
        "final_answer","final_status"
    ]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Created: {OUT_CSV}")
    print(f"Created: {OUT_DIR}")
    print("Checked Q33, Q34, Q53, Q103, Q143 plus final-row physical-circle evidence.")
    print("No scanner/recognition files were modified.")

if __name__ == "__main__":
    main()
