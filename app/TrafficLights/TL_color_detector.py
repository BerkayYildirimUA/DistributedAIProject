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





class TinyTrafficLightNet(nn.Module):
    """
    CNN for 3x64x32 inputs.
    Two conv blocks + global average pooling + linear.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # 3 x 64 x 32 -> 16 x 32 x 16
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 16 x 32 x 16 -> 32 x 16 x 8
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 32 x 16 x 8 -> 32 x 1 x 1
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)        # [B, 32, 1, 1]
        x = x.view(x.size(0), -1)   # [B, 32]
        x = self.classifier(x)      # [B, num_classes]
        return x


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
        self.classes = ["green", "red", "yellow"]

        # device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        # model setup
        self.model = self.load_tl_model("app/TrafficLights/Models/traffic_light_classifier16.pth", self.device)
        #self.model = self.make_model(num_classes=len(self.classes)).to(self.device)
        #self._load_weights(ckpt_path)
        self.model.eval()

        self.input_h, self.input_w = 64, 32       # crop size used by the classifier
        self.use_tracking = False    # not used here, but keeps attribute parity
        self.last_track_ids = torch.empty(0, dtype=torch.long)  # parity
        self.conf_default = 0.0      # unused, kept for parity
        self.nms_iou = 0.0           # unused, kept for parity

        # Inference transform (must match training normalization)
        self.infer_tf = transforms.Compose([
            transforms.Resize((self.input_h, self.input_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


    # --- structure parity helper ---
    def preprocess_frame(self, frame: np.ndarray):
        """Return frame plus (w,h) like ObjectDetector.preprocess_frame."""
        frame_w, frame_h = frame.shape[1], frame.shape[0]
        return frame, frame_w, frame_h



#------------------------- Specify model
    # @staticmethod
    # def make_model(num_classes: int =3):
    #     m = torchvision.models.resnet18(weights=None)  # "IMAGENET1K_V1", !!!!!!!!!!! weights=None at runtime
    #     m.fc = nn.Linear(m.fc.in_features,
    #                      num_classes)  # we replace the orignal 1000 output by 3 (traffic light colors)
    #     return m

    @staticmethod
    def make_model(num_classes: int = 3):
        return TinyTrafficLightNet(num_classes=num_classes)

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
            # If no weights, leave randomly initialized
            pass

    def classify_crop_hsv(self, crop_bgr: np.ndarray):
        """
        Heuristic HSV-based color classification for a single crop.
        Returns (label:str | None, score:float)
        """
        # BGR -> HSV  (OpenCV: H in [0,179], S,V in [0,255])
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    
        lower_red1 = np.array([0, 80, 80], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([160, 80, 80], dtype=np.uint8)
        upper_red2 = np.array([179, 255, 255], dtype=np.uint8)
    
        # Yellow: around H=30 in OpenCV
        lower_yellow = np.array([15, 60, 60], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
    
        # Green: around H=60 in OpenCV, with some slack
        lower_green = np.array([35, 40, 40], dtype=np.uint8)
        upper_green = np.array([90, 255, 255], dtype=np.uint8)
    
        # Masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
        red_count    = cv2.countNonZero(mask_red)
        yellow_count = cv2.countNonZero(mask_yellow)
        green_count  = cv2.countNonZero(mask_green)
    
        print(
            "H range:", hsv[..., 0].min(), hsv[..., 0].max(),
            "S range:", hsv[..., 1].min(), hsv[..., 1].max(),
            "V range:", hsv[..., 2].min(), hsv[..., 2].max(),
        )
    
        print("counts R/Y/G:", red_count, yellow_count, green_count)
    
        counts = np.array([green_count, yellow_count, red_count], dtype=np.int32)
        labels = ["green", "yellow", "red"]
    
        best_idx = int(np.argmax(counts))
        best_count = counts[best_idx]
        total_colored = counts.sum()
    
        # Ignore tiny noise
        MIN_PIXELS = 5
        if best_count < MIN_PIXELS or total_colored == 0:
            return None, 0.0
    
        #  confidence: share of dominant color among all colored pixels
        score = float(best_count) / float(total_colored)
        return labels[best_idx], score


    @torch.no_grad()
    def predict_colors_batch(self,
                    frame_bgr: np.ndarray,                      #frame of the camera (H, W, 3)
                    boxes_xyxy: torch.Tensor | None = None,
                    pad_ratio: float = 0.02):                   #adds padding around the box when cropping
        """
        returns: list of (label, confidence) for each input box

        Parameters:
        - frame_bgr: np.ndarray HxWx3 (OpenCV BGR), full frame in OpenCV BGR format
        - boxes_xyxy: torch.Tensor Nx4 (x1,y1,x2,y2) in pixel coordinates x1 = left border, x2 = right border, y1 = top border, y2 = bottom border. If None or empty, then it
            returns empty tensors
        - pad_ratio: float, padding ratio around each box before cropping
        """

        # Handle for no inputs
        if boxes_xyxy is None or boxes_xyxy.numel() == 0:
            empty_boxes = torch.empty((0, 4), dtype=torch.float32)
            empty_scores = torch.empty((0,), dtype=torch.float32)
            overall_conf = {cls: 0.0 for cls in self.classes}
            return empty_boxes, [], empty_scores, None

        # Convertion from PIL to RGB
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        W, H = pil_img.size

        crops = []
        out_boxes = []

        # make sure tensor on CPU for index ops
        boxes_cpu = boxes_xyxy.detach().cpu()
        for (x1, y1, x2, y2) in boxes_cpu.tolist():
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # padding around bbox for robustness
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
            empty_boxes = torch.empty((0, 4), dtype=torch.float32)
            empty_scores = torch.empty((0,), dtype=torch.float32)
            overall_conf = {}
            for cls in self.classes:
                overall_conf[cls] = 0.0
            return empty_boxes, [], empty_scores, None

        x = torch.stack(crops, dim=0).to(self.device)   # [N, 3, H, W]
        logits = self.model(x)                          # [N, C]
        probs = logits.softmax(1).detach().cpu()        # [N, C] # Nx3, forward pass + get probabilities + moving back to cpu for processing
        conf, idx = probs.max(dim=1)                    # N, conf = max probability of a class, idx = class with the highest probability
        class_ids = idx.to(torch.long)
        scores = conf.to(torch.float32)
        out_boxes = torch.tensor(out_boxes, dtype=torch.float32)

        # ---- per-TL color labels ----
        tl_colors = []
        for i in class_ids:
            tl_colors.append(self.classes[int(i)])

        # deciding global color
        # Sum of class probabilities over all relevant TLs
        total = probs.sum(dim=0)   # tells us how much total prob mass each color has across all TLs
        total_sum = float(total.sum())      # total sum across all total mass probs
        if total_sum > 0.0:
            global_probs = total / total_sum  # normalized [C]
            best_idx = int(global_probs.argmax().item())
            overall_color = self.classes[best_idx]  #  "red", "green", "yellow"
        else:
            overall_color = None

        return out_boxes,tl_colors, scores, overall_color












