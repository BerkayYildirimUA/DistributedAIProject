# pip install torch torchvision
import torch, torch.nn as nn, torchvision
from PIL import Image
# data loading
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2



# --- Classes you want ---
CLASSES = ["red", "yellow", "green"]

# Model (ResNet18 head)
def make_model(num_classes=len(CLASSES)):
    m = torchvision.models.resnet18(weights=None)       #"IMAGENET1K_V1", !!!!!!!!!!! weights=None at runtime
    m.fc = nn.Linear(m.fc.in_features, num_classes)         #we replace the orignal 1000 output by 3 (traffic light colors)
    return m

# at runtime
def load_tl_model(ckpt_path, device):
    model = make_model().to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


# Transforms
train_tf = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
# inference transform (must match training normalization)
INFER_TF = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


train_ds = datasets.ImageFolder("data/train", transform=train_tf)
val_ds   = datasets.ImageFolder("data/val",   transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0, pin_memory=True)





# Training sketch
def train_one_epoch(model, loader, opt, device, loss_fn):
    model.train()
   # ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        #loss = ce(logits, y)
        loss = loss_fn(logits, y)

        loss.backward()
        opt.step()
        loss_sum += loss.item()*x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_model(model, loader, device, loss_fn):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
       # loss = ce(logits, y)
        loss = loss_fn(logits, y)
        loss_sum += loss.item()*x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total



#training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# choose between interfence with loaded or train
model = load_tl_model("traffic_light_classifier.pth", device)   # !!!!!!!!! for interfence
# model = make_model().to(device)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

# if yellow is slightly fewer, give it a small boost weight
weights = torch.tensor([1.0, 1.1, 1.0], device=device)  # [red, yellow, green]
ce = nn.CrossEntropyLoss(weight=weights)


for epoch in range(15):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device, ce)
    va_loss, va_acc = eval_model(model, val_loader, device, ce)
    print(f"Epoch {epoch:02d} | train_acc={tr_acc:.3f} | val_acc={va_acc:.3f}")

torch.save(model.state_dict(), "traffic_light_classifier.pth")







# Tiny dataset for <image, bbox> → crop
class TLStateDataset(torch.utils.data.Dataset):
    def __init__(self, rows, transform):
        """
        rows: list of dicts with { "img_path": str, "bbox": [x1,y1,x2,y2], "label": str }
        """
        self.rows = rows
        self.t = transform
        self.class_to_idx = {c:i for i,c in enumerate(CLASSES)}         #color names to index
    def __len__(self):
        return len(self.rows)                   #returns how many samples are in the dataset (for epoch end)
    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["img_path"]).convert("RGB")
        x1,y1,x2,y2 = r["bbox"]
        # optional padding: 10%     to add background
        w, h = img.size
        pw, ph = int(0.1*(x2-x1)), int(0.1*(y2-y1))
        x1p, y1p = max(0, x1-pw), max(0, y1-ph)
        x2p, y2p = min(w, x2+pw), min(h, y2+ph)
        crop = img.crop((x1p, y1p, x2p, y2p))
        crop = self.t(crop)                             #applies the transformation (96×96, color jitter, normalization) + turns into torch with shape: [3, 96, 96]
        y = self.class_to_idx[r["label"]]               #converts color name into index
        return crop, y                                  # returns the image tensor and the numeric class label



# Inference utility (given full image + bbox)
# @torch.no_grad()
# def predict_state(model, pil_image, bbox, device="cpu", conf=False):
#     x1,y1,x2,y2 = bbox
#     crop = pil_image.crop((x1,y1,x2,y2)).resize((96,96))
#     t = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])         #channel-wise mean and standard deviation of the ImageNet dataset (which was used to pretrain ResNet18)
#     x = t(crop).unsqueeze(0).to(device)
#     logits = model(x)
#     p = logits.softmax(1).squeeze(0).cpu()          # converts the raw scores into probabilities that sum to 1, squeeze for 1D, and move tensor from GPU to CPU
#     cls_idx = int(p.argmax().item())                # find the index with the highest value
#     return (CLASSES[cls_idx], float(p[cls_idx])) if conf else CLASSES[cls_idx]      #if conf is true we send





@torch.no_grad()
def predict_colors_batch(model, frame_bgr, boxes_xyxy, device, pad_ratio=0.10):
    """
    frame_bgr: np.ndarray HxWx3 (OpenCV BGR)
    boxes_xyxy: torch.Tensor Nx4 (x1,y1,x2,y2) in pixel coords
    returns: list of (label, confidence) for each input box
    """
    if boxes_xyxy.numel() == 0:
        return []

    # Convert frame to PIL RGB once (cheaper than per-crop)
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    W, H = pil_img.size

    crops = []
    for (x1, y1, x2, y2) in boxes_xyxy.tolist():
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # optional padding around bbox for robustness
        pw, ph = int(pad_ratio * (x2 - x1)), int(pad_ratio * (y2 - y1))
        x1p, y1p = max(0, x1 - pw), max(0, y1 - ph)
        x2p, y2p = min(W, x2 + pw), min(H, y2 + ph)
        crop = pil_img.crop((x1p, y1p, x2p, y2p))
        crops.append(INFER_TF(crop))

    x = torch.stack(crops, dim=0).to(device)
    probs = model(x).softmax(1).cpu()  # Nx3
    conf, idx = probs.max(dim=1)       # N
    labels = [CLASSES[i] for i in idx.tolist()]
    return list(zip(labels, conf.tolist()))
