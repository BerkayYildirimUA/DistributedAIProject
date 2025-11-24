import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------------- CONFIG ----------------
IMG_SIZE    = 128
BATCH_TRAIN = 64
BATCH_VAL   = 128
EPOCHS      = 5
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_OUT = Path("Models/traffic_light_classifier3.pth")
META_OUT = Path("Models/traffic_light_classifier_meta3.json")

TRAIN_DIR = Path("data/train")
VAL_DIR   = Path("data/val")

# ---------------- DATA ----------------
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tf)
val_ds   = datasets.ImageFolder(str(VAL_DIR),   transform=val_tf)




# ----- compute class weights from training set -----
# import torch
#
# class_to_idx = train_ds.class_to_idx          # e.g. {'green':0,'red':1,'yellow':2}
# targets = torch.tensor(train_ds.targets)      # ImageFolder provides .targets
# num_classes = 3
# counts = torch.bincount(targets, minlength=num_classes).float()
#
# # Inverse frequency as a baseline
# inv_freq = 1.0 / torch.clamp(counts, min=1)
# cls_weights = inv_freq / inv_freq.mean()      # normalize around 1.0
#
# # Optional: explicitly boost yellow a bit more
# yellow_idx = class_to_idx["yellow"]
# cls_weights[yellow_idx] *= 1.05                # tune 1.2–2.0 as needed
#
# print("class_to_idx:", class_to_idx)
# print("counts:", counts.tolist())
# print("class_weights:", cls_weights.tolist())

#----------------------------------------------------------------




# Windows tip: keep num_workers=0 unless wrapped in if __name__ == "__main__"
pin = (DEVICE == "cuda")
train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,
                          num_workers=0, pin_memory=pin)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_VAL, shuffle=False,
                          num_workers=0, pin_memory=pin)

# Save class mapping/metadata for inference
META_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(META_OUT, "w") as f:
    json.dump({"class_to_idx": train_ds.class_to_idx,
               "img_size": IMG_SIZE}, f, indent=2)

# ---------------- MODEL ----------------
# torchvision >= 0.13: use weights enum
# try:
#     weights = models.ResNet18_Weights.IMAGENET1K_V1
#     model = models.resnet18(weights=weights)
# except Exception:
#     # fallback for older versions
#     model = models.resnet18(pretrained=True)
#
# model.fc = nn.Linear(model.fc.in_features, 3)
# model.to(DEVICE)
#
# criterion = nn.CrossEntropyLoss()
# #criterion = nn.CrossEntropyLoss(weight=cls_weights.to(DEVICE))
#
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#
# scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))


# ---------------- MODEL ----------------
from torchvision.models import shufflenet_v2_x0_5

model = shufflenet_v2_x0_5(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.to(DEVICE)

# Loss (use class weights if you computed them)
criterion = nn.CrossEntropyLoss()  # or nn.CrossEntropyLoss(weight=cls_weights.to(DEVICE))

# Optimizer + AMP
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))





# ---------------- TRAIN ----------------
best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = total = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()

    train_loss = running_loss / max(1, total)
    train_acc = correct / max(1, total)

    # -------- VALIDATE --------
    model.eval()
    v_correct = v_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            v_correct += (logits.argmax(1) == y).sum().item()
            v_total += y.numel()
    val_acc = v_correct / max(1, v_total)

    print(f"[{epoch:02d}/{EPOCHS}] train_loss={train_loss:.4f} "
          f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), str(CKPT_OUT))
        print(f"  ↳ saved best to {CKPT_OUT} (val_acc={best_val_acc:.3f})")

print("Done.")

