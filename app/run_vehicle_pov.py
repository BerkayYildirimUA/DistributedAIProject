import cv2
import numpy as np
from sympy.physics.units import velocity

import constants

from data_processors.lane_detector import LaneDetector
from data_processors.intersection_detector import IntersectionDetector
import torch
from TrafficLights.TL_color_detector import TL_color_detector
from data_processors.object_detector import ObjectDetector
from data_processors.object_distance_calculator import ObjectDistanceCalculator
from memory.shared_memory import (
    RGBCameraMemory, DepthCameraMemory, VehicleDistanceMemory, VehicleStateMemory, LaneTubeMemory, RadarMemory, CameraCalibrationMemory
)
from data_processors.motion_tubes import MotionTubeProjector
from engine.pov_visualiser import POVVisualiser
from data_processors.radar_points_projector import RadarPointsProjector

# Attach to shared memory
rgb_camera_memory = RGBCameraMemory().get_read_access()
depth_camera_memory = DepthCameraMemory().get_read_access()
#vehicle_distance_memory = VehicleDistanceMemory().get_write_access()
radar_memory = RadarMemory().get_read_access()
camera_calibration_memory = CameraCalibrationMemory().get_read_access()

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


radar_points_projector = RadarPointsProjector()
try:
    import time

    while True:
        # Convert to Torch tensor and normalize
        frame = rgb_camera_memory.read()
        depth_map = depth_camera_memory.read()
        radar_data = radar_memory.read()
        if np.count_nonzero(frame) == 0:
            continue
        # Detect objects
        boxes, class_ids, scores = object_detector.get_objects(frame)
        # Get distance for each object
        # distances=object_distance_calculator.get_distances(boxes,depth_map)
        cam_mats = camera_calibration_memory.read()
        K = np.asarray(cam_mats[0, :3, :3], dtype=np.float64)
        P = np.asarray(cam_mats[1], dtype=np.float64)

        # projection of radar_points to camera
        u, v, z, Pc, kept, idx = RadarPointsProjector.project_radar_points_world_to_image(
            radar_data, K, P, constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT
        )

        vel = radar_data[idx, 4].astype(np.float64)  # radial velocity

        distances = object_distance_calculator.get_radar_distances(
            boxes,
            (u, v, z, Pc, kept),
            radar_data,
            idx
        )

        # vehicle state
        speed_ms, steer_rad = state_memory.read()
        # init tube_projector once we know frame size
        # MOTION TUBES
        lanes = tube_projector.get_projected_lanes(float(speed_ms), float(steer_rad))
        # VISION MODEL
        # lanes = lane_detector.get_lanes(frame,int_degree=3)

        # Lanes
        # get also trajectory
        lane_1 = [tuple(p[0]) for p in lanes[0]]
        lane_2 = [tuple(p[0]) for p in lanes[2]]
        lane_1_x = np.array([x[0] for x in lane_1])
        lane_2_x = np.array([x[0] for x in lane_2])
        min_lane_distance = abs(max(lane_1_x)-min(lane_2_x))
        is_intersected=intersection_detector.is_intersecting_list_trajectory_based(boxes,lanes[1],min_lane_distance/2,0.1*min_lane_distance)
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
        tl_boxes_colored, tl_colors, tl_scores = tl_color_detector.predict_colors_batch(frame, tl_boxes)

        # Visualise
        visualiser = POVVisualiser(
            object_detector.classes,
            frame,
            boxes,
            class_ids,
            scores,
            distances,
            is_intersected,
            lanes,
            [tl_boxes_colored, tl_colors, tl_scores]
        )

        visualiser.overlay_radar_points(
            projection=(u, v, z, Pc, kept),
            velocities=vel
        )

        visualiser.show()

        # if len(lanes) > 0:
        #     bird_eye_visualiser.show(boxes,class_ids,lanes)

finally:
    cv2.destroyAllWindows()


