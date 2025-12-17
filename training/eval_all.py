import re
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import csv

#This script automates evaluation across multiple training runs inside runs/detect. The key goal is reproducible comparison:
#if I trained several variants (different learning rates, freeze settings, epochs, image sizes),
#this script creates one summary file instead of manually checking each run.

DATA = "data.yaml"
RUNS_DIR = Path("runs/detect")
OUT_DIR = Path("runs/eval_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def infer_imgsz_from_name(name, default=640):
    # Searches "..._640imgsize_..." or "..._832imgsize_..."
    m = re.search(r"_(\d{3,4})imgsize", name)
    return int(m.group(1)) if m else default

def to_float(x):
    try:
        #
        arr = np.array(x)
        return float(arr.mean())
    except Exception:
        return float(x)

def main():
    rows = []
    model_paths = sorted(RUNS_DIR.glob("*/weights/best.pt"))

    for mp in model_paths:
        run_name = mp.parents[1].name  # map name of the run
        imgsz = infer_imgsz_from_name(run_name, default=640)

        print(f"\n=== Evaluating {run_name} (imgsz={imgsz}) ===")
        model = YOLO(str(mp))
        metrics = model.val(
            data=DATA,
            imgsz=imgsz,
            split="val",
            conf=0.001,
            iou=0.7,
            plots=False,
            save_json=False,
            verbose=False,
        )
        row = {
            "run": run_name,
            "imgsz": imgsz,
            "map50_95": to_float(metrics.box.map),     # mAP@[.5:.95]
            "map50":    to_float(metrics.box.map50),
            "map75":    to_float(metrics.box.map75),
            "precision":to_float(metrics.box.p),
            "recall":   to_float(metrics.box.r),
        }
        rows.append(row)

    # sorteer op mAP50-95 aflopend
    rows.sort(key=lambda r: r["map50_95"], reverse=True)

    # schrijf CSV
    csv_path = OUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved table → {csv_path}")

    # barplot (mAP50-95)
    labels = [r["run"] for r in rows]
    vals = [r["map50_95"] for r in rows]
    plt.figure(figsize=(10, 0.6*len(labels)))
    y = np.arange(len(labels))
    plt.barh(y, vals)
    plt.yticks(y, labels)
    plt.xlabel("mAP@0.50:0.95")
    plt.title("Model comparison")
    plt.tight_layout()
    plot_path = OUT_DIR / "map50_95_bar.png"
    plt.savefig(plot_path, dpi=200)
    print(f"Saved plot → {plot_path}")

    # print compacte tabel in console
    print("\n=== SUMMARY ===")
    for r in rows:
        print(f"{r['run']:>45s}  imgsz={r['imgsz']:4d}  mAP50-95={r['map50_95']:.3f}  "
              f"mAP50={r['map50']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}")

if __name__ == "__main__":
    main()
