import time

import numpy as np
import cv2
import torch
import torchvision
from ultralytics import YOLO
import os

# https://medium.com/@zain.18j2000/how-to-use-your-yolov11-model-with-onnx-runtime-69f4ea243c01
# TODO: remove overlapping boxes
class ObjectDetection:
    def __init__(self):
        # Initialize model
        print("CUDA:", torch.cuda.is_available())
        self.model = YOLO("best.pt")
        self.classes = ["Vehicle", "Motor", "Bike","traffic light","traffic sign","pedestrian"]
        self.input_size = 800
        self.laneDetection = LaneDetection()

    # Convert frame to correct input format for yolo
    def preprocess_frame(self,frame):
        frame_w, frame_h = frame.shape[1], frame.shape[0]
        # frame = cv2.resize(frame, (self.input_size, self.input_size))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = frame.transpose(2, 0, 1)
        # frame = frame.reshape(1, 3, self.input_size, self.input_size)
        # frame = frame/255.0
        # frame=frame.astype(np.float32)

        return frame, frame_w, frame_h

    def get_objects(self, frame, frame_w, frame_h, conf_threshold=0.1):
        # Run prediction on GPU
        results = self.model.predict(source=frame, device="cuda", conf=conf_threshold, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            # No detections
            return torch.empty((0, 4)), torch.empty((0,), dtype=torch.long), torch.empty((0,))

        # Extract predictions
        boxes_xyxy = results[0].boxes.xyxy.cpu()  # shape: (N, 4)
        scores = results[0].boxes.conf.cpu()  # shape: (N,)
        class_ids = results[0].boxes.cls.cpu().long()  # shape: (N,)

        # Scale boxes if needed (Ultralytics outputs original image coordinates)
        # scaling_kernel = torch.tensor([frame_w, frame_h, frame_w, frame_h]) / self.input_size
        scaling_kernel=torch.tensor([1, 1, 1, 1])
        boxes_scaled = boxes_xyxy * scaling_kernel

        # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
        keep_indices = torchvision.ops.nms(boxes_scaled, scores, iou_threshold=0.5)

        return boxes_scaled[keep_indices], class_ids[keep_indices], scores[keep_indices]


    # Draw boxes onto the frame
    def add_object_boxes(self, frame, boxes, scores, class_ids):
        for (x1,y1,x2,y2), score, cls_id in zip(boxes, class_ids, scores):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_name = self.classes[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame


    def detect_and_add_overlay(self, frame, depth_map=None):
        start_time = time.time()
        pro_frame, frame_w, frame_h = self.preprocess_frame(frame)
        boxes,class_ids,scores = self.get_objects(pro_frame, frame_w, frame_h)

        distance_vehicle_in_front_m = 0
    
        for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_name = self.classes[int(cls_id)]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
            # calculate distance_m = the closest distance within the bounding box relative to the ego car.
            if depth_map is not None:
                # clip coords to image dimensions
                x1d = max(0, min(depth_map.shape[1]-1, x1))
                x2d = max(0, min(depth_map.shape[1]-1, x2))
                y1d = max(0, min(depth_map.shape[0]-1, y1))
                y2d = max(0, min(depth_map.shape[0]-1, y2))
                crop = depth_map[y1d:y2d+1, x1d:x2d+1]
                if crop.size > 0:
                    distance_m = np.nanmin(crop)
                    cv2.putText(frame, f"{distance_m:.1f} m", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if (self.vehicle_is_in_front(x1, y1, x2, y2, frame)):
                        distance_vehicle_in_front_m = distance_m

        frame_with_boxes_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print(f"Inference + overlay time: {(time.time()-start_time)*1000:.2f} ms")
        return frame_with_boxes_bgr, distance_vehicle_in_front_m

    #TODO: check whether car is in front of the ego vehicle using Image Processing techniques.
    def vehicle_is_in_front(self,  x1, y1, x2, y2, frame=None):
        return True