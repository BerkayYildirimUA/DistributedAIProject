import numpy as np
import torch
import torchvision
from ultralytics import YOLO
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import app.constants as constants
# https://medium.com/@zain.18j2000/how-to-use-your-yolov11-model-with-onnx-runtime-69f4ea243c01
# this class is the 'front-end' of the computer vision module, it takes one RGB frame coming from CARLA and
# returns bounding boxes, class IDs and confidence scores for all detected objects. And because tracking is enabled,
# it tries to keep the same ID across frames instead of being treated as a brand new object every frame.

class ObjectDetector:
    def __init__(self, use_tracking = True):
        # Initialize model
        print("CUDA:", torch.cuda.is_available())   # check if GPU used
        self.model = YOLO("app/resources/best6.pt") # load the trained model
        self.classes = constants.OBJECT_CLASS_NAMES # storing the class names from constants
        self.input_size = 640

        # tracking
        self.use_tracking = use_tracking
        self.tracker_cfg = "bytetrack.yaml"         # ultralytics built-in tracker
        self.conf_default = 0.15                    # confidence threshold

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.last_track_ids = torch.empty(0, dtype=torch.long) # This is where you store the track IDs from the last processed frame.
                                                               # If there are no detections, it stays empty.

    # Convert frame to correct input format for yolo
    def preprocess_frame(self,frame):
        frame_w, frame_h = frame.shape[1], frame.shape[0]
        return frame, frame_w, frame_h

    # this main method is called every frame
    def get_objects(self, frame, conf_threshold=0.15):
        # use class default if None was passed
        conf = self.conf_default if conf_threshold is None else conf_threshold

        # Detect if tracking enabled
        if self.use_tracking:
            results = self.model.track(     # this runs YOLO detection on the frame and then runs ByteTrack to associate detections with previous frame detections and assign IDs
                source=frame,
                device=self.device,
                conf=conf,                  # this filters out low-confidence detections
                iou=0.3,
                persist=True,               # this tells Ultralytics to keep the tracker state across frames so ID's stay consistent over time
                tracker=self.tracker_cfg,   # bytetrack chosen here
                verbose=False,              # don't print all those ouputs/logs (timings, tracker info, preprocess/inference speeds...)
            )
        else:
            results = self.model.predict(   # if tracking is disabled, we just use pure detection: boxes + classes + confidences and no IDs
                source=frame,
                device=self.device,
                conf=conf,
                verbose=False,
            )

        if len(results) == 0 or len(results[0].boxes) == 0:         # if nothing found then return empty tensors
            self.last_track_ids = torch.empty(0, dtype=torch.long)
            # No detections
            return torch.empty((0, 4)), torch.empty((0,), dtype=torch.long), torch.empty((0,))

        # Extract predictions
        boxes_xyxy = results[0].boxes.xyxy.cpu()  # shape: (N, 4)
        scores = results[0].boxes.conf.cpu()  # shape: (N,)
        class_ids = results[0].boxes.cls.cpu().long()  # shape: (N,)

        # Track IDs (only exist in track mode)
        ids = results[0].boxes.id
        ids = ids.cpu().long() if ids is not None else torch.empty((0,), dtype=torch.long)
        self.last_track_ids = ids

        return boxes_xyxy, class_ids, scores
