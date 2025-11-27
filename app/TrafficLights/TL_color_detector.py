# pip install torch torchvision
from __future__ import annotations

import numpy as np
import torch, torch.nn as nn, torchvision
from PIL import Image
# data loading
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import cv2

from torchvision.models import shufflenet_v2_x0_5


# tl_color_trainer.py
import torch, torchvision
from torch import nn



class TL_color_detector:
    """
    Color classifier for traffic-light crops, shaped to mirror ObjectDetector's class structure.
    - __init__: loads a ResNet18-based classifier (3 classes: red, yellow, green)
    - preprocess_frame: returns (frame, frame_w, frame_h) for interface symmetry
    - get_objects: returns (boxes_xyxy, class_ids, scores)
        * boxes_xyxy: the same boxes you pass in (Nx4, xyxy pixel coords)
        * class_ids: color indices: 0=red, 1=yellow, 2=green
        * scores: confidence (softmax prob of predicted class)
    """
    def __init__(self,
                 ckpt_path: str = "traffic_light_classifier2.pth",
                 device: str | None = None):
        self.classes = ["green", "yellow", "red"]

        # device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        # model setup
        self.model = self.load_tl_model("app/TrafficLights/Models/traffic_light_classifier2.pth", self.device)
        #self.model = self.make_model(num_classes=len(self.classes)).to(self.device)
        #self._load_weights(ckpt_path)
        self.model.eval()

        # Match fields present in ObjectDetector for easier drop-in
        self.input_size = 128         # crop size used by the classifier
        self.use_tracking = False    # not used here, but keeps attribute parity
        self.last_track_ids = torch.empty(0, dtype=torch.long)  # parity
        self.conf_default = 0.0      # unused, kept for parity
        self.nms_iou = 0.0           # unused, kept for parity

        # Inference transform (must match training normalization)
        self.infer_tf = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


    # --- structure parity helper ---
    def preprocess_frame(self, frame: np.ndarray):
        """Return frame plus (w,h) like ObjectDetector.preprocess_frame."""
        frame_w, frame_h = frame.shape[1], frame.shape[0]
        return frame, frame_w, frame_h

    # @staticmethod
    # Model (ResNet18 head)
    def make_model(num_classes: int =3):
        m = torchvision.models.resnet18(weights=None)  # "IMAGENET1K_V1", !!!!!!!!!!! weights=None at runtime
        m.fc = nn.Linear(m.fc.in_features,
                         num_classes)  # we replace the orignal 1000 output by 3 (traffic light colors)
        return m

    # @staticmethod
    # def make_model(num_classes: int = 3):
    #     m = shufflenet_v2_x0_5(weights=None)  # keep None; you load your own ckpt
    #     m.fc = nn.Linear(m.fc.in_features, num_classes)
    #     return m

    @classmethod
    # at runtime
    def load_tl_model(cls, ckpt_path, device):
        model = cls.make_model().to(device)                 #cls refers to the class TL_color_detector
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
        model.eval()
        return model

    def _load_weights(self, ckpt_path: str):
        try:
            state = torch.load(ckpt_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                self.model.load_state_dict(state["state_dict"])
            else:
                self.model.load_state_dict(state)
        except FileNotFoundError:
            # If no weights, leave randomly initialized; still functional API.
            pass


    @torch.no_grad()
    def predict_colors_batch(self,
                    frame_bgr: np.ndarray,                      #frame of the camera (H, W, 3)
                    boxes_xyxy: torch.Tensor | None = None,
                    conf_threshold: float | None = None,        #filters out low-confidence predictions (not used yet)
                    pad_ratio: float = 0.02):                   #adds padding around the box when cropping
        """
        frame_bgr: np.ndarray HxWx3 (OpenCV BGR)
        boxes_xyxy: torch.Tensor Nx4 (x1,y1,x2,y2) in pixel coordinates x1 = left border, x2 = right border, y1 = top border, y2 = bottom border
        returns: list of (label, confidence) for each input box

        Parameters:

        - frame_bgr: np.ndarray, full frame in OpenCV BGR format
        - boxes_xyxy: torch.Tensor or None, Nx4 tensor with (x1,y1,x2,y2) in pixel coordinates. If None or empty, then it
            returns empty tensors
        - conf_threshold: float or None, optional confidence filter
        - pad_ratio: float, padding ratio around each box before cropping
        """

        # Handle no inputs
        if boxes_xyxy is None or boxes_xyxy.numel() == 0:
            return (torch.empty((0, 4), dtype=torch.float32),
                    torch.empty((0,), dtype=torch.long),
                    torch.empty((0,), dtype=torch.float32))

        # Convert once to PIL RGB
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        W, H = pil_img.size

        crops = []
        out_boxes = []

        # make sure tensor on CPU for index ops
        boxes_cpu = boxes_xyxy.detach().cpu()
        for (x1, y1, x2, y2) in boxes_cpu.tolist():
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # optional padding around bbox for robustness
            pw, ph = int(pad_ratio * (x2 - x1)), int(pad_ratio * (y2 - y1))
            x1p, y1p = max(0, x1 - pw), max(0, y1 - ph)
            x2p, y2p = min(W, x2 + pw), min(H, y2 + ph)
            if x2p <= x1p or y2p <= y1p:
                # skips degenerate bounding formats
                continue
            crop = pil_img.crop((x1p, y1p, x2p, y2p))
            crops.append(self.infer_tf(crop))
            out_boxes.append([x1, y1, x2, y2])
        if len(crops) == 0:
            return (torch.empty((0, 4), dtype=torch.float32),
                    torch.empty((0,), dtype=torch.long),
                    torch.empty((0,), dtype=torch.float32))

        x = torch.stack(crops, dim=0).to(self.device)  # N,3,96,96
        probs = self.model(x).softmax(1).detach().cpu()  # Nx3, forward pass + get probabilities + moving back to cpu for processing
        # conf, idx = probs.max(dim=1)  # N
        # labels = [CLASSES[i] for i in idx.tolist()]
        # return list(zip(labels, conf.tolist()))
        conf, idx = probs.max(dim=1)                    # N, conf = max probability of a class, idx = class with the highest probability
        class_ids = idx.to(torch.long)
        scores = conf.to(torch.float32)
        out_boxes = torch.tensor(out_boxes, dtype=torch.float32)

        # Optional confidence filter (kept for parity with ObjectDetector)
        if conf_threshold is not None:
            keep = scores >= float(conf_threshold)
            out_boxes = out_boxes[keep]
            class_ids = class_ids[keep]
            scores = scores[keep]
        tl_colors = [self.classes[int(i)] for i in class_ids]

        return out_boxes,tl_colors, scores








# --- Classes you want ---
CLASSES = ["red", "yellow", "green"]






# Transforms
train_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
# inference transform (must match training normalization)
INFER_TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])







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



# #training loop
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # choose between interfence with loaded or train
# model = load_tl_model("traffic_light_classifier.pth", device)   # !!!!!!!!! for interfence
# # model = make_model().to(device)
#
# opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
#
# # if yellow is slightly fewer, give it a small boost weight
# weights = torch.tensor([1.0, 1.1, 1.0], device=device)  # [red, yellow, green]
# ce = nn.CrossEntropyLoss(weight=weights)
#
#
# for epoch in range(15):
#     tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device, ce)
#     va_loss, va_acc = eval_model(model, val_loader, device, ce)
#     print(f"Epoch {epoch:02d} | train_acc={tr_acc:.3f} | val_acc={va_acc:.3f}")
#
# torch.save(model.state_dict(), "traffic_light_classifier.pth")







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






