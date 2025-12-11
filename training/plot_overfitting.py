from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RUN_DIR = Path("runs/detect/trainm_all_640_s")  # pas aan naar jouw run
csv = RUN_DIR / "results.csv"
df = pd.read_csv(csv)

df["train_total"] = df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]
df["val_total"]   = df["val/box_loss"]   + df["val/cls_loss"]   + df["val/dfl_loss"]

best_epoch = int(df["val_total"].idxmin())
print(f"Beste val (totale loss) op epoch: {best_epoch}")

plt.figure()
plt.plot(df["epoch"], df["train_total"], label="train_total")
plt.plot(df["epoch"], df["val_total"],   label="val_total")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Total loss")
plt.grid(True)
out = RUN_DIR / "total_loss.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
