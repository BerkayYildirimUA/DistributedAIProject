import json
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------------- CONFIG ----------------
IMG_SIZE    = 64
IMG_H = 96 # 64
IMG_W = 96 # 32
BATCH_TRAIN = 64
BATCH_VAL   = 128
EPOCHS      = 10
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_OUT = Path("Models/traffic_light_classifier6.pth")
META_OUT = Path("Models/traffic_light_classifier_meta6.json")

TEST_DIR = Path("data/test")
TRAIN_DIR = Path("data/train")
VAL_DIR   = Path("data/val")

# ---------------- DATA preprocessing ----------------
train_tf = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
 #   transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
  #  transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# same preprocessing as validation
test_tf = val_tf


train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tf)
val_ds   = datasets.ImageFolder(str(VAL_DIR),   transform=val_tf)
test_ds = datasets.ImageFolder(str(TEST_DIR), transform=test_tf)


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


# For Windows: keep num_workers=0 or otherwise wrapping in if __name__ == "__main__"
pin = (DEVICE == "cuda")
train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,
                          num_workers=0, pin_memory=pin)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_VAL, shuffle=False,
                          num_workers=0, pin_memory=pin)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_VAL,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE == "cuda"),
)

# Save class mapping/metadata for inference
META_OUT.parent.mkdir(parents=True, exist_ok=True)
# with open(META_OUT, "w") as f:
#     json.dump({"class_to_idx": train_ds.class_to_idx,
#                "img_h": IMG_H,
#                "img_w": IMG_W}, f, indent=2)

model_meta = {
    "type": "custom_cnn",
    "input_shape": [3, IMG_H, IMG_W],
    "description": "3 conv blocks + 2 fully-connected layers",
}

meta = {
    "class_to_idx": train_ds.class_to_idx,     # ImageFolder uses {'green':0,'red':1,'yellow':2}
    "img_h": IMG_H,
    "img_w": IMG_W,
    "model": model_meta,
}

with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)



# ---------------- MODEL ----------------
# try:
#     weights = models.ResNet18_Weights.IMAGENET1K_V1
#     model = models.resnet18(weights=weights)
# except Exception:
#     # fallback for older versions
#     model = models.resnet18(pretrained=True)
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
# from torchvision.models import shufflenet_v2_x0_5
#
# model = shufflenet_v2_x0_5(weights=None)
# model.fc = nn.Linear(model.fc.in_features, 3)
# model.to(DEVICE)
#
# # Loss (use class weights if you computed them)
# criterion = nn.CrossEntropyLoss()  # or nn.CrossEntropyLoss(weight=cls_weights.to(DEVICE))
#
# # Optimizer + AMP
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))



# ---------------- MODEL ----------------
# class TrafficLightNet(nn.Module):
#     """
#     CNN for 3x64x32 inputs
#     64x32 -> 32x16 -> 16x8 -> 8x4 feature maps
#     """
#     def __init__(self, num_classes: int = 3):
#         super().__init__()
#         self.features = nn.Sequential(
#             # 3 x 64 x 32 -> 32 x 32 x 16
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),
#
#             # 32 x 32 x 16 -> 64 x 16 x 8
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),
#
#             # 64 x 16 x 8 -> 128 x 8 x 4
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),
#         )
#
#         # After 3x MaxPool(2): 128 -> 64 -> 32 -> 16
#         # so feature map = 128 channels, 16x16 spatial
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(128 * 8 * 4, 256),  # flattened size = 128 * 8 * 4 = 4096 -> 256
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)          # [B, 128, 8, 4]
#         x = x.view(x.size(0), -1)     # [B, 4096]
#         x = self.classifier(x)        # [B, num_classes]
#         return x
#
# num_classes = len(train_ds.classes)  # should be 3: red/green/yellow
# model = TrafficLightNet(num_classes=num_classes).to(DEVICE)
#
# # Loss (use class weights here if you want)
# criterion = nn.CrossEntropyLoss()
#
# # Optimizer + AMP
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))


#----------------------Model--------------------------------------
class TrafficLightNet2(nn.Module):
    """
    Tiny CNN for 3x64x32 inputs
    64x32 -> 32x16 -> 16x8 feature maps (only 2 blocks)
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 x 64 x 32 -> 16 x 32 x 16
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 16 x 32 x 16 -> 24 x 16 x 8
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # After 2x MaxPool(2): 64x32 -> 32x16 -> 16x8
        flat_dim = 24 * 16 * 8  # 24 channels, 16x8 spatial = 3072

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(flat_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)          # [B, 24, 16, 8]
        x = x.view(x.size(0), -1)     # [B, 3072]
        x = self.classifier(x)        # [B, num_classes]
        return x

num_classes = len(train_ds.classes)  # should be 3: red/green/yellow
model = TrafficLightNet2(num_classes=num_classes).to(DEVICE)

# Loss (use class weights here if you want)
criterion = nn.CrossEntropyLoss()

# Optimizer + AMP
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))


# ---------------- TRAIN ----------------
best_val_acc = 0.0
train_losses = []
val_losses = []

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
    train_losses.append(train_loss)     # <--- store train loss

    # -------- VALIDATE --------
    model.eval()
    v_running_loss = 0.0
    v_correct = v_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            v_loss = criterion(logits, y)
            v_running_loss += v_loss.item() * x.size(0)

            v_correct += (logits.argmax(1) == y).sum().item()
            v_total += y.numel()

    val_loss = v_running_loss / max(1, v_total)
    val_acc = v_correct / max(1, v_total)
    val_losses.append(val_loss)

    print(f"[{epoch:02d}/{EPOCHS}] "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), str(CKPT_OUT))

print("Done.")

# -------- PLOT LOSSES --------
epochs_range = range(1, EPOCHS + 1)

plt.figure()
plt.plot(epochs_range, train_losses, label="Train loss")
plt.plot(epochs_range, val_losses, label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Models/traffic_light_loss_curve.png")  # optional
plt.show()


# ----------------------------------------------------
# ---------------- TEST ----------------
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import time


# Load metadata to get consistent class mapping
with open(META_OUT, "r") as f:
    meta = json.load(f)
class_to_idx = meta["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

num_classes = len(class_to_idx)

# Recreate model and load best weights
#model = TrafficLightNet(num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load(CKPT_OUT, map_location=DEVICE))
model.eval()

# ---- Run evaluation on test set ----
all_preds = []
all_targets = []
correct = 0
total = 0
total_infer_time = 0.0  # seconds (only forward pass)
total_images = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        # --- measure forward-pass time ---
        start = time.perf_counter()
        logits = model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()  # make sure all kernels are finished
        end = time.perf_counter()
        # -------------------------------

        batch_time = end - start
        batch_size = x.size(0)
        total_infer_time += batch_time
        total_images += batch_size

        preds = logits.argmax(1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

        correct += (preds == y).sum().item()
        total += y.numel()

test_acc = correct / max(1, total)
print(f"\n=== Test set performance ===")
print(f"Test accuracy: {test_acc:.3f} ({correct}/{total})")


# ---- Inference speed stats ----
if total_images > 0 and total_infer_time > 0:
    ms_per_image = (total_infer_time / total_images) * 1000.0
    print(f"\n=== Inference speed ===")
    print(f"Average time per image:            {ms_per_image:.3f} ms/image")

#----- Per class metrics:
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

print("\nClassification report:")
print(
    classification_report(
        all_targets,
        all_preds,
        target_names=[idx_to_class[i] for i in range(num_classes)],
        digits=3,
    )
)

cm = confusion_matrix(all_targets, all_preds)
print("\nConfusion matrix (rows = true, cols = predicted):")
print(cm)