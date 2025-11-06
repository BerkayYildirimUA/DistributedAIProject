import cv2
import numpy as np
import torch
from TrafficLights import TL_color_detector
from data_processors.object_detector import ObjectDetector
from data_processors.object_distance_calculator import ObjectDistanceCalculator
from memory.shared_memory import RGBCameraMemory, DepthCameraMemory, VehicleDistanceMemory
from engine.pov_visualiser import POVVisualiser

# Attach to shared memory
rgb_camera_memory = RGBCameraMemory().get_read_access()
depth_camera_memory = DepthCameraMemory().get_read_access()
vehicle_distance_memory = VehicleDistanceMemory().get_write_access()

object_detector = ObjectDetector()
object_distance_calculator=ObjectDistanceCalculator()
try:
    import time
    while True:
        # Convert to Torch tensor and normalize
        frame=rgb_camera_memory.read()
        depth_map = depth_camera_memory.read()
        if np.count_nonzero(frame) == 0:
            # No data yet, skip this iteration
            continue
        # Detect objects
        boxes, class_ids, scores =object_detector.get_objects(frame)

        # Filter detections to only traffic lights
        if len(class_ids) > 0:
            # Build a boolean mask using class names
            cls_names = [object_detector.classes[int(c)] for c in class_ids.tolist()]
            is_tl = torch.tensor("traffic light", dtype=torch.bool)

            tl_boxes = boxes[is_tl]
        else:
            tl_boxes = torch.empty((0, 4))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tl_model = TL_color_detector.load_tl_model("traffic_light_classifier.pth", device)

        # Classify colors for those TL boxes
        tl_preds = TL_color_detector.predict_colors_batch(tl_model, frame, tl_boxes, device)  # list of (label, conf)

        # (Optional) draw labels on the frame
        for (box, (label, conf)) in zip(tl_boxes.tolist(), tl_preds):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

        # Get distance for each object
        distances=object_distance_calculator.get_distances(boxes,depth_map)

        # Visualise
        visualiser= POVVisualiser(
            object_detector.classes,
            frame,boxes,
            class_ids,
            scores,
            distances)
        visualiser.show()

finally:
    cv2.destroyAllWindows()


