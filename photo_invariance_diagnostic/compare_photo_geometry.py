
from pathlib import Path
import csv, math, statistics, sys

OPTIONS = ["A","B","C","D"]

def read_report(path):
    rows, qfinal = {}, {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            q = int(r["question"])
            opt = r["option"].strip()
            rows[(q,opt)] = r
            qfinal[q] = {
                "answer": r.get("final_answer",""),
                "status": r.get("final_status",""),
            }
    return rows, qfinal

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_photo_geometry.py report_472.csv report_480.csv")
        raise SystemExit(2)

    p1, p2 = Path(sys.argv[1]), Path(sys.argv[2])
    r1, f1 = read_report(p1)
    r2, f2 = read_report(p2)

    detail = []
    for q in range(1,181):
        for opt in OPTIONS:
            a,b = r1.get((q,opt)), r2.get((q,opt))
            if not a or not b:
                continue
            x1,y1 = float(a["crop_center_x"]), float(a["crop_center_y"])
            x2,y2 = float(b["crop_center_x"]), float(b["crop_center_y"])
            dx,dy = x2-x1,y2-y1
            detail.append({
                "question":q,
                "column":(q-1)//45+1,
                "row_in_column":(q-1)%45+1,
                "option":opt,
                "x_report1":x1,"y_report1":y1,
                "x_report2":x2,"y_report2":y2,
                "dx":dx,"dy":dy,
                "distance_px":math.hypot(dx,dy),
                "answer_report1":f1[q]["answer"],
                "status_report1":f1[q]["status"],
                "answer_report2":f2[q]["answer"],
                "status_report2":f2[q]["status"],
                "decision_changed":(
                    f1[q]["answer"] != f2[q]["answer"]
                    or f1[q]["status"] != f2[q]["status"]
                ),
            })

    changes=[]
    for q in range(1,181):
        if (
            f1[q]["answer"] != f2[q]["answer"]
            or f1[q]["status"] != f2[q]["status"]
        ):
            ds=[r for r in detail if r["question"]==q]
            changes.append({
                "question":q,
                "column":(q-1)//45+1,
                "row_in_column":(q-1)%45+1,
                "answer_report1":f1[q]["answer"],
                "status_report1":f1[q]["status"],
                "answer_report2":f2[q]["answer"],
                "status_report2":f2[q]["status"],
                "mean_center_shift_px":round(statistics.mean(r["distance_px"] for r in ds),3),
                "max_center_shift_px":round(max(r["distance_px"] for r in ds),3),
                "mean_dx":round(statistics.mean(r["dx"] for r in ds),3),
                "mean_dy":round(statistics.mean(r["dy"] for r in ds),3),
            })

    def write_csv(name, rows):
        with open(name,"w",encoding="utf-8",newline="") as f:
            if rows:
                w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    write_csv("center_drift_all_options.csv", detail)
    write_csv("decision_changes.csv", changes)

    print("Changed decisions:", len(changes))
    print("Created center_drift_all_options.csv")
    print("Created decision_changes.csv")

if __name__ == "__main__":
    main()
