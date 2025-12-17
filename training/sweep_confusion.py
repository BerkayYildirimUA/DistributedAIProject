import numpy as np
from ultralytics import YOLO

DATA = "data.yaml"
MODEL = "runs/detect/train_large_832imgsize_0.01lr_300epochs_16batch/weights/best.pt"

PROJECT_SWEEP = "runs/sweeps"    # thresholds sweep folders
PROJECT_PLOTS = "runs/sweeps2"   # only the best plots (confusion matrix, PR, ...)

def f1_score(p, r):
    return 2 * p * r / (p + r)

def to_scalar(x):
    arr = np.array(x)
    return float(arr.mean())

def main():
    model = YOLO(MODEL)
    thr_grid = np.round(np.linspace(0.05, 0.85, 17), 3)

    best = {"thr": None, "f1": -1.0, "p": 0.0, "r": 0.0}

    for thr in thr_grid:
        print(f"conf={thr:.3f}")
        m = model.val(
            data=DATA,
            imgsz=832,
            split="val",
            conf=float(thr),
            iou=0.7,
            plots=False,
            verbose=False,
            project=PROJECT_SWEEP,
            name=f"tmp_{int(thr*100):02d}",
            save_json=False
        )

        p = to_scalar(m.box.p)
        r = to_scalar(m.box.r)
        f1 = f1_score(p, r)

        if f1 > best["f1"]:
            best = {"thr": float(thr), "f1": f1, "p": p, "r": r}

    print(f"\nBest threshold by F1: conf={best['thr']:.3f}  "
          f"P={best['p']:.3f}  R={best['r']:.3f}  F1={best['f1']:.3f}")

    out_name = f"best_conf_{int(best['thr']*100):02d}"
    model.val(
        data=DATA,
        imgsz=832,
        split="test",
        conf=best["thr"],
        iou=0.7,
        plots=True,             # confusion matrix + curves
        verbose=False,
        project=PROJECT_PLOTS,  # saves in runs/....
        name=out_name,
        save_json=False
    )

    print(f"Saved plots in {PROJECT_PLOTS}/{out_name}")

if __name__ == "__main__":
    main()
