import cv2
import numpy as np
import app.constants as constants

from app.data_processors.intersection_detector import IntersectionDetector
import torch
from app.TrafficLights.TL_color_detector import TL_color_detector
from app.data_processors.object_detector import ObjectDetector
from app.data_processors.object_distance_calculator import ObjectDistanceCalculator
from app.memory.shared_memory import (
    RGBCameraMemory, FrameIdMemory, VehicleDistanceMemory, VehicleStateMemory, LaneTubeMemory, RadarMemory, CameraCalibrationMemory
)
from app.data_processors.motion_tubes import MotionTubeProjector
from app.engine.pov_visualiser import POVVisualiser
from app.data_processors.radar_points_projector import RadarPointsProjector
from app.data_processors.metrics_logger import MetricsLogger
from app.data_processors.detected_classes_count import detected_classes_count

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
detected_classes_count = detected_classes_count()
estimated_object_count_metrics_logger = MetricsLogger(constants.ESTIMATED_OBJECT_IN_FRONT_COUNT_FILE, compress=True)
estimated_tl_count_metrics_logger = MetricsLogger(constants.ESTIMATED_TRAFFIC_LIGHT_COUNT_FILE, compress=True)
estimated_ts_count_metrics_logger = MetricsLogger(constants.ESTIMATED_TRAFFIC_SIGN_COUNT_FILE, compress=True)
estimated_vehicle_count_metrics_logger = MetricsLogger(constants.ESTIMATED_VEHICLE_COUNT_FILE, compress=True)
estimated_pedestrian_count_metrics_logger = MetricsLogger(constants.ESTIMATED_PEDESTRIAN_COUNT_FILE, compress=True)

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
            if d < closest_vehicle_distance:
                closest_vehicle_distance = d

        vehicle_distance_memory.write(closest_vehicle_distance)

        cls_names = []
        if len(class_ids) > 0:
            for c in class_ids:  # we convert the number values to the class names
                cls_names.append(object_detector.classes[int(c)])

        # Traffic lights selecting out of all the recognized objects in the current frame
        if abs(steer_rad) < 0.15:                #to avoid that the car detects non-relevant traffic lights during a turn on intersection
            if len(class_ids) > 0:                  # if there are objects detected present
                mask = []
                for n in cls_names:                         #creating a mask for traffic light class
                    if n == "traffic light":
                        mask.append(True)
                    else:
                        mask.append(False)
                is_tl = torch.tensor(mask, dtype=torch.bool, device=boxes.device)

                tl_boxes = boxes[is_tl]         #filtering and selecting only the boxes of traffic lights



                # Apply ROI filter (to only select the relevant traffic lights for our car)
                H, W = frame.shape[0], frame.shape[1]
                #       x ∈ [0.45W, 0.61W] and y ≥ 0.35
                LEFT_RATIO = 0.45           # borders that match for town 5
                RIGHT_RATIO = 0.67
                Y_MIN_RATIO = 0.35
                x_min = LEFT_RATIO * W
                x_max = RIGHT_RATIO * W
                y_min = Y_MIN_RATIO * H
                keep_mask = []
                for x1, y1, x2, y2 in tl_boxes.tolist():
                    cx = 0.5 * (x1 + x2)                    #we calculate the centers
                    cy = 0.5 * (y1 + y2)

                    in_roi = (x_min <= cx <= x_max) and (cy >= y_min)
                    keep_mask.append(in_roi)

                keep_mask = torch.tensor(keep_mask, dtype=torch.bool, device=boxes.device)
                tl_boxes = tl_boxes[keep_mask]          # with the mask we filter the relevant traffic lights
            else:
                tl_boxes = torch.empty((0, 4))
        else:
            tl_boxes = torch.empty((0, 4))              #avoid that it prints random tl boxes while taking a turn

        # Perform the color classification
        tl_boxes_colored, tl_colors, tl_scores, overall_conf = tl_color_detector.predict_colors_batch(frame, tl_boxes)
        print("Global color distribution:", overall_conf)

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


        # TODO: if distance contains nan values, this does not work. We are comparing object counts while relying on distances
        #       that are not the same (radar vs ground truth), therefore containing two layers of errors. Difference in python
        #       environments makes this difficult though
        dist_arr = np.asarray(distances, dtype=np.float64)

        # Build mask of boxes that are within a valid distance
        valid_distance_mask = np.isfinite(dist_arr) & (
                dist_arr <= constants.MAX_OBJECT_DETECT_DISTANCE
        )

        # Turn mask into indices
        valid_indices = np.nonzero(valid_distance_mask)[0]

        # How many objects are in front (within distance)
        estimated_objects_in_front = int(valid_indices.size)

        # Filter boxes based on the distance mask
        if valid_indices.size > 0:
            # Filter boxes
            if isinstance(boxes, torch.Tensor):
                valid_indices_t = torch.from_numpy(valid_indices).to(boxes.device)
                filtered_boxes = boxes[valid_indices_t]
            else:
                filtered_boxes = boxes[valid_distance_mask]

            # Filter cls_names in exactly the same way
            filtered_cls_names = [cls_names[i] for i in valid_indices]
        else:
            # No valid objects: empty boxes + empty names
            if isinstance(boxes, torch.Tensor):
                filtered_boxes = boxes[:0]  # shape [0, 4]
            else:
                filtered_boxes = boxes[0:0]
            filtered_cls_names = []

        # Count classes only on filtered boxes
        counted_classes = detected_classes_count.count_objects(filtered_boxes, filtered_cls_names)

        counted_vehicles = counted_classes["vehicles"]
        counted_pedestrians = counted_classes["pedestrians"]
        counted_traffic_lights = counted_classes["traffic_lights"]
        counted_traffic_signs = counted_classes["traffic_signs"]
        total_counted = counted_classes["total"]
        print("===========")
        print(counted_vehicles)
        print(counted_pedestrians)
        print(counted_traffic_lights)
        print(counted_traffic_signs)
        print("===========")


        estimated_object_count_metrics_logger.log(
            frame_id=frame_id,
            estimated_yolo_objects=total_counted,
        )

        estimated_ts_count_metrics_logger.log(
            frame_id=frame_id,
            estimated_traffic_signs=counted_traffic_signs
        )

        estimated_tl_count_metrics_logger.log(
            frame_id=frame_id,
            estimated_traffic_lights=counted_traffic_lights
        )

        estimated_vehicle_count_metrics_logger.log(
            frame_id=frame_id,
            estimated_vehicles_front_count=counted_vehicles,
        )

        estimated_pedestrian_count_metrics_logger.log(
            frame_id=frame_id,
            estimated_pedestrians=counted_pedestrians
        )
finally:
    try:
        estimated_object_count_metrics_logger.close()
        estimated_tl_count_metrics_logger.close()
        estimated_ts_count_metrics_logger.close()
        estimated_vehicle_count_metrics_logger.close()
        estimated_pedestrian_count_metrics_logger.close()
        print("Loggers closed in new_env")
    except Exception as e:
        print(f"Error closing loggers: {e}")
    cv2.destroyAllWindows()


