import cv2
import numpy as np

import app.constants as constants

from app.data_processors.lane_detector import LaneDetector
from app.data_processors.intersection_detector import IntersectionDetector
import torch
from app.TrafficLights.TL_color_detector import TL_color_detector
from app.data_processors.object_detector import ObjectDetector
from app.data_processors.object_distance_calculator import ObjectDistanceCalculator
from app.memory.shared_memory import (
    RGBCameraMemory, DepthCameraMemory, VehicleDistanceMemory, VehicleStateMemory, LaneTubeMemory, RadarMemory, CameraCalibrationMemory
)
from app.data_processors.motion_tubes import MotionTubeProjector
from app.engine.pov_visualiser import POVVisualiser
from app.data_processors.radar_points_projector import RadarPointsProjector
from app.data_processors.metrics_logger import MetricsLogger

# Attach to shared memory
rgb_camera_memory = RGBCameraMemory().get_read_access()
#depth_camera_memory = DepthCameraMemory().get_read_access()
vehicle_distance_memory = VehicleDistanceMemory().get_write_access()
radar_memory = RadarMemory().get_read_access()
camera_calibration_memory = CameraCalibrationMemory().get_read_access()
frame_id_memory = FrameIdMemory().get_read_access()

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
estimated_object_count_metrics_logger = MetricsLogger(constants.ESTIMATED_OBJECT_IN_FRONT_COUNT_FILE, compress=True)

try:
    import time

    while True:
        frame_id = int(frame_id_memory.read()[0])
        # Convert to Torch tensor and normalize
        frame = rgb_camera_memory.read()
        #depth_map = depth_camera_memory.read()
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
        if len(lane_1_x) ==0 or len(lane_2_x) == 0:
            is_intersected=[False]*len(boxes)
        else:
            min_lane_distance = abs(max(lane_1_x)-min(lane_2_x))
            is_intersected=intersection_detector.is_intersecting_list_trajectory_based(boxes,lanes[1],min_lane_distance/2,0.1*min_lane_distance)
        # is_intersected=intersection_detector.is_intersecting_list(lanes[0],lanes[1],boxes)

        # Write distance of closest vehicle in lane to shared memory
        intersecting_box_indices = [i for i, x in enumerate(is_intersected) if x]
        closest_vehicle_distance = np.inf
        for indice in intersecting_box_indices:
            d = distances[indice]
            if not np.isfinite(d) or d > constants.MAX_LEAD_ACTOR_DISTANCE:
                continue
            if d < closest_vehicle_distance:
                closest_vehicle_distance = d

        vehicle_distance_memory.write(closest_vehicle_distance)


        # Only Traffic lights selecting
        if len(class_ids) > 0:
            cls_names = [object_detector.classes[int(c)] for c in class_ids.tolist()]
            is_tl = torch.tensor([n == "traffic light" for n in cls_names],
                                 dtype=torch.bool, device=boxes.device)
            tl_boxes = boxes[is_tl]
        else:
            tl_boxes = torch.empty((0, 4))

        # Perform the color classification
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


        dist_arr = np.asarray(distances, dtype=np.float64)

        # TODO: if distance contains nan values, this does not work. We are comparing object counts while relying on distances
        #       that are not the same (radar vs ground truth), therefore containing two layers of errors. Difference in python
        #       environments makes this difficult though
        valid_distance_mask = np.isfinite(dist_arr) & (dist_arr <= constants.MAX_OBJECT_DETECT_DISTANCE)

        estimated_objects_in_front = int(np.sum(valid_distance_mask))

        estimated_object_count_metrics_logger.log(
            frame_id=frame_id,
            estimated_yolo_objects=estimated_objects_in_front,
        )


finally:
    try:
        estimated_object_count_metrics_logger.close()
        print("Loggers closed in new_env")
    except Exception as e:
        print(f"Error closing loggers: {e}")
    cv2.destroyAllWindows()


