import cv2
import numpy as np

import app.constants as constants
from app.data_processors.sign_classifier import SignClassifier

from app.data_processors.lane_detector import LaneDetector
from app.data_processors.intersection_detector import IntersectionDetector
import torch
from app.TrafficLights.TL_color_detector import TL_color_detector
from app.data_processors.object_detector import ObjectDetector
from app.data_processors.object_distance_calculator import ObjectDistanceCalculator
from app.memory.shared_memory import (
    RGBCameraMemory, DepthCameraMemory, VehicleDistanceMemory, VehicleStateMemory, LaneTubeMemory, RadarMemory,
    CameraCalibrationMemory, TrafficSignMemory, TrafficLightMemory, TrafficLightDistanceMemory
)
from app.data_processors.motion_tubes import MotionTubeProjector
from app.engine.pov_visualiser import POVVisualiser
from app.data_processors.radar_points_projector import RadarPointsProjector

# Attach to shared memory
rgb_camera_memory = RGBCameraMemory().get_read_access()
#depth_camera_memory = DepthCameraMemory().get_read_access()
vehicle_distance_memory = VehicleDistanceMemory().get_write_access()
radar_memory = RadarMemory().get_read_access()
camera_calibration_memory = CameraCalibrationMemory().get_read_access()

object_detector = ObjectDetector()
state_memory = VehicleStateMemory().get_read_access()
traffic_sign_memory=TrafficSignMemory().get_write_access()
traffic_light_memory=TrafficLightMemory().get_write_access()
traffic_light_distance_memory=TrafficLightDistanceMemory().get_write_access()
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

sign_classifier = SignClassifier()

radar_points_projector = RadarPointsProjector()
try:
    import time

    while True:
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



        # Traffic lights selecting out of all the recognized objects in the current frame
        if abs(steer_rad) < 0.15:                #to avoid that the car detects non-relevant traffic lights during a turn on intersection
            if len(class_ids) > 0:                  # if there are objects detected present
                cls_names = []
                for c in class_ids:                         #we convert the number values to the class names
                    cls_names.append(object_detector.classes[int(c)])

                mask = []
                for n in cls_names:                         #creating a mask for traffic light class
                    if n == "traffic light":
                        mask.append(True)
                    else:
                        mask.append(False)
                is_tl = torch.tensor(mask, dtype=torch.bool, device=boxes.device)


                boxes_indexes=np.array(range(len(boxes)))
                tl_boxes = boxes[is_tl]         #filtering and selecting only the boxes of traffic lights
                tl_indexes = boxes_indexes[is_tl]



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
                tl_indexes = tl_indexes[keep_mask]
            else:
                tl_boxes = torch.empty((0, 4))
                tl_indexes = torch.empty((0, 0))

        else:
            tl_boxes = torch.empty((0, 4))              #avoid that it prints random tl boxes while taking a turn
            tl_indexes = torch.empty((0, 0))

        # Get minimal traffic light distance
        tl_min_distance = min(np.array(distances)[tl_indexes])
        print(distances, tl_indexes,tl_min_distance)
        traffic_light_distance_memory.write(tl_min_distance)



        # Perform the color classification
        tl_boxes_colored, tl_colors, tl_scores, overall_conf = tl_color_detector.predict_colors_batch(frame, tl_boxes)
        # print("Global color distribution:", overall_conf)
        print(overall_conf)
        tl_color = max(overall_conf, key=overall_conf.get)
        tr_color_index= constants.TL_COLOR_TO_INDEX[tl_color]
        traffic_light_memory.write(tr_color_index)

        traffic_sign = sign_classifier.signal_classifier(frame, boxes, class_ids)
        traffic_sign_memory.write(traffic_sign)
        # if not (traffic_signs == -1):
        #     print(traffic_signs)

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


