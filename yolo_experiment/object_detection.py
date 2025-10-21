import time

import numpy as np
import cv2
import torch
import torchvision
from ultralytics import YOLO
import os
from lane_detection import LaneDetection

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

    def visualize_lanes(self, frame):
        """TODO: used for debugging, remove this! """
        nwindows = 9
        margin = 100
        minpix = 50

        # Warp frame to bird-eye view
        img_warped = self.laneDetection.get_perspective_matrices(frame)

        # Convert to grayscale and threshold
        img_gray = cv2.cvtColor(img_warped, cv2.COLOR_RGB2GRAY)
        _, img_binary = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY)

        # Extract lane feature pixels
        nonzerox, nonzeroy, window_height = self.laneDetection.extract_features(img_binary, nwindows)

        # Find lane pixels
        leftx, lefty, rightx, righty, out_img = self.laneDetection.find_lane_pixels(
            img_binary, nwindows, margin, minpix, nonzerox, nonzeroy, window_height
        )

        # Fit polynomials and get visualization overlay
        lane_overlay = self.laneDetection.fit_poly(img_binary, leftx, lefty, rightx, righty)

        # Warp the overlay back to original perspective
        return lane_overlay

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


    def detect_and_add_overlay(self, frame, depth_map=None, show_lanes=True):
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
        if show_lanes:
            lane_overlay = self.visualize_lanes(frame)
            frame = cv2.addWeighted(frame, 0.7, lane_overlay, 0.3, 0)


        frame_with_boxes_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print(f"Inference + overlay time: {(time.time()-start_time)*1000:.2f} ms")
        return frame_with_boxes_bgr, distance_vehicle_in_front_m


    #TODO: check whether car is in front of the ego vehicle using Image Processing techniques.
    def vehicle_is_in_front(self,  x1, y1, x2, y2, frame=None):
       """  if frame is None:
            return False;

        nwindows = 9
        margin = 100
        minpix = 50

        # warp frame to birds-eye view
        img_warped = self.laneDetection.get_perspective_matrices(frame)

        # convert to grayscale such that lane pixels are white
        img_gray = cv2.cvtColor(img_warped, cv2.COLOR_RGB2GRAY)
        _, img_binary = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY)

        # Extract lane feature pixels
        nonzerox, nonzeroy, window_height = self.laneDetection.extract_features(img_binary, nwindows)

        # Find left/right lane pixels
        leftx, lefty, rightx, righty, _ = self.laneDetection.find_lane_pixels(
            img_binary, nwindows, margin, minpix, nonzerox, nonzeroy, window_height
        )

        # Fit polynomials
        out_img = self.laneDetection.fit_poly(img_binary, leftx, lefty, rightx, righty)

        # Get polynomial coefficients for left and right lanes
        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 else [0, 0, 0]
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 else [0, 0, 0]

        # Check if vehicle bounding box is inside the detected lane
        ys = np.linspace(y1, y2, num=5)
        left_xs = left_fit[0]*ys**2 + left_fit[1]*ys + left_fit[2]
        right_xs = right_fit[0]*ys**2 + right_fit[1]*ys + right_fit[2]

        for lx, rx in zip(left_xs, right_xs):
            if x1 >= lx and x2 <= rx:
                return True
        return False """

        return True