import numpy as np
import torch
import torchvision
from ultralytics import YOLO
# https://medium.com/@zain.18j2000/how-to-use-your-yolov11-model-with-onnx-runtime-69f4ea243c01

class ObjectDetector:
    def __init__(self, use_tracking = True):
        # Initialize model
        print("CUDA:", torch.cuda.is_available())
        self.model = YOLO("resources/best.pt")
        self.classes = ["Vehicle", "Motor", "Bike","traffic light","traffic sign","pedestrian"]
        # TODO: create model with smaller input size
        self.input_size = 640

        # tracking
        self.use_tracking = use_tracking
        self.tracker_cfg = "bytetrack.yaml" # ultralytics built-in tracker
        self.conf_default = 0.25
        self.nms_iou = 0.5

        # EMA smoothing per track-id
        self.ema_beta = 0.70  # 0 = no smoothing; closer to 1 = more smoothing (slower to react)
        self.track_history = {}  # id -> np.array([x1,y1,x2,y2])

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.last_track_ids = torch.empty(0, dtype=torch.long)
        # -------------------------

    # --- small helper for smoothing ---
    def _ema_smooth(self, boxes_xyxy: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        if boxes_xyxy.numel() == 0 or ids is None or ids.numel() == 0:
            return boxes_xyxy
        out = []
        for i in range(boxes_xyxy.shape[0]):
            box = boxes_xyxy[i].cpu().numpy()
            tid = int(ids[i].item())
            if tid in self.track_history:
                prev = self.track_history[tid]
                box = self.ema_beta * prev + (1.0 - self.ema_beta) * box
            self.track_history[tid] = box
            out.append(box)
        return torch.tensor(np.stack(out), dtype=boxes_xyxy.dtype)

    # Convert frame to correct input format for yolo
    def preprocess_frame(self,frame):
        frame_w, frame_h = frame.shape[1], frame.shape[0]
        return frame, frame_w, frame_h

    def get_objects(self, frame, conf_threshold=0.1):
        # use class default if None was passed
        conf = self.conf_default if conf_threshold is None else conf_threshold

        # Detect (with tracking if enabled)
        if self.use_tracking:
            results = self.model.track(
                source=frame,
                device=self.device,
                conf=conf,
                iou=0.5,
                persist=True,  # keep identities over frames
                tracker=self.tracker_cfg,
                verbose=False,
            )
        else:
            results = self.model.predict(
                source=frame,
                device=self.device,
                conf=conf,
                verbose=False,
            )

        if len(results) == 0 or len(results[0].boxes) == 0:
            self.last_track_ids = torch.empty(0, dtype=torch.long)
            # No detections
            return torch.empty((0, 4)), torch.empty((0,), dtype=torch.long), torch.empty((0,))

        # Extract predictions
        boxes_xyxy = results[0].boxes.xyxy.cpu()  # shape: (N, 4)
        scores = results[0].boxes.conf.cpu()  # shape: (N,)
        class_ids = results[0].boxes.cls.cpu().long()  # shape: (N,)

        # tracker IDs (if available)
        ids = results[0].boxes.id
        ids = ids.cpu().long() if ids is not None else None

        # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
        keep_indices = torchvision.ops.nms(boxes_xyxy, scores, iou_threshold=self.nms_iou)
        boxes_xyxy = boxes_xyxy[keep_indices]
        scores = scores[keep_indices]
        class_ids = class_ids[keep_indices]

        # Optional EMA smoothing if we have track IDs
        if ids is not None:
            ids = ids[keep_indices]
            boxes_xyxy = self._ema_smooth(boxes_xyxy, ids)
            self.last_track_ids = ids
        else:
            self.last_track_ids = torch.empty(0, dtype=torch.long)

        return boxes_xyxy, class_ids, scores
