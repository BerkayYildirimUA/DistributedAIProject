import torch
import torchvision
from ultralytics import YOLO
# https://medium.com/@zain.18j2000/how-to-use-your-yolov11-model-with-onnx-runtime-69f4ea243c01

class ObjectDetector:
    def __init__(self):
        # Initialize model
        print("CUDA:", torch.cuda.is_available())
        self.model = YOLO("resources/best.pt")
        self.classes = ["Vehicle", "Motor", "Bike","traffic light","traffic sign","pedestrian"]
        # TODO: create model with smaller input size
        self.input_size = 800

    # Convert frame to correct input format for yolo
    def preprocess_frame(self,frame):
        frame_w, frame_h = frame.shape[1], frame.shape[0]
        return frame, frame_w, frame_h

    def get_objects(self, frame, conf_threshold=0.1):
        # Run prediction on GPU
        results = self.model.predict(source=frame, device="cuda", conf=conf_threshold, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            # No detections
            return torch.empty((0, 4)), torch.empty((0,), dtype=torch.long), torch.empty((0,))

        # Extract predictions
        boxes_xyxy = results[0].boxes.xyxy.cpu()  # shape: (N, 4)
        scores = results[0].boxes.conf.cpu()  # shape: (N,)
        class_ids = results[0].boxes.cls.cpu().long()  # shape: (N,)

        # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
        keep_indices = torchvision.ops.nms(boxes_xyxy, scores, iou_threshold=0.5)

        return boxes_xyxy[keep_indices], class_ids[keep_indices], scores[keep_indices]


