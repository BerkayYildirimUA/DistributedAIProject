import cv2
import numpy as np

from data_processors.lane_detector import LaneDetector
from data_processors.intersection_detector import IntersectionDetector
import torch
from TrafficLights.TL_color_detector import TL_color_detector
from data_processors.object_detector import ObjectDetector
from data_processors.object_distance_calculator import ObjectDistanceCalculator
from memory.shared_memory import (
    RGBCameraMemory, DepthCameraMemory, VehicleDistanceMemory, VehicleStateMemory, LaneTubeMemory
)
from data_processors.motion_tubes import MotionTubeProjector
from engine.pov_visualiser import POVVisualiser

# Attach to shared memory
rgb_camera_memory = RGBCameraMemory().get_read_access()
depth_camera_memory = DepthCameraMemory().get_read_access()
vehicle_distance_memory = VehicleDistanceMemory().get_write_access()

object_detector = ObjectDetector()
state_memory = VehicleStateMemory().get_read_access()
lane_mem = LaneTubeMemory(max_pts=256).get_write_access()
object_distance_calculator=ObjectDistanceCalculator()
tube_projector = MotionTubeProjector(
    img_w=640, img_h=480,
    fov_deg=90.0,  # CARLA RGB camera default
    cam_height=1.5,  # jouw camera z=1.5
    lane_width=3.6,
    wheelbase=2.8,
    meters_ahead=40.0,
    center_offset_m=0.0  # evt. +0.2 of -0.2 afstellen
)
# bird_eye_visualiser=BirdVisualiser(640,480)
intersection_detector=IntersectionDetector()
tl_color_detector = TL_color_detector()
# lane_detector=LaneDetector()


try:
    import time
    while True:
        # Convert to Torch tensor and normalize
        frame = rgb_camera_memory.read()
        depth_map = depth_camera_memory.read()
        if np.count_nonzero(frame) == 0:
            continue

        # vehicle state
        speed_ms, steer_rad = state_memory.read()
        # init tube_projector once we know frame size
        # MOTION TUBES
        lanes = tube_projector.get_projected_lanes(float(speed_ms), float(steer_rad))
        # VISION MODEL
        # lanes = lane_detector.get_lanes(frame,int_degree=3)

        # Detect + distances
        boxes, class_ids, scores = object_detector.get_objects(frame)
        distances = object_distance_calculator.get_distances(boxes, depth_map)

        # Lanes
        # get also trajectory
        is_intersected=intersection_detector.is_intersecting_list_trajectory_based(boxes,lanes[1],3.6/2,0)
        # is_intersected=intersection_detector.is_intersecting_list(lanes[0],lanes[1],boxes)

        # --- Stage 2: select only traffic lights ---
        if len(class_ids) > 0:

            cls_names = [object_detector.classes[int(c)] for c in class_ids.tolist()]
            is_tl = torch.tensor([n == "traffic light" for n in cls_names],
                                 dtype=torch.bool, device=boxes.device)
            tl_boxes = boxes[is_tl]
        else:
            tl_boxes = torch.empty((0, 4))

        # --- Stage 3: classify traffic light colors ---
        if len(tl_boxes) > 0:
            tl_boxes_colored, tl_color_ids, tl_scores = tl_color_detector.predict_colors_batch(frame, tl_boxes)
            tl_colors = [tl_color_detector.classes[int(i)] for i in tl_color_ids]
            for (box, color, conf) in zip(tl_boxes_colored.tolist(), tl_colors, tl_scores.tolist()):
                x1, y1, x2, y2 = map(int, box)
                color_map = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[color], 2)

                # place TL color ABOVE the class label (which is at y1-5)
                label = f"{color} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                y_top = max(0, y1 - 5 - th - 4)  # a bit higher than the class text

                cv2.rectangle(frame, (x1, y_top - th - 4), (x1 + tw + 4, y_top), (0, 0, 0), -1)
                cv2.putText(frame, f"{color} {conf:.2f}", (x1, max(0, y_top)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            tl_boxes_colored = torch.empty((0, 4))
            tl_color_ids = torch.empty((0,))
            tl_scores = torch.empty((0,))

        # Visualise
        visualiser = POVVisualiser(
            object_detector.classes,
            frame,
            boxes,
            class_ids,
            scores,
            distances,
            is_intersected,
            lanes
        )

        visualiser.show()

        # if len(lanes) > 0:
        #     bird_eye_visualiser.show(boxes,class_ids,lanes)

finally:
    cv2.destroyAllWindows()


