import cv2
import numpy as np
import constants

from data_processors.object_detector import ObjectDetector
from data_processors.object_distance_calculator import ObjectDistanceCalculator
from memory.shared_memory import RGBCameraMemory, DepthCameraMemory, VehicleDistanceMemory, RadarMemory, CameraCalibrationMemory
from engine.pov_visualiser import POVVisualiser

# Attach to shared memory
rgb_camera_memory = RGBCameraMemory().get_read_access()
depth_camera_memory = DepthCameraMemory().get_read_access()
#vehicle_distance_memory = VehicleDistanceMemory().get_write_access()
radar_memory = RadarMemory().get_read_access()
camera_calibration_memory = CameraCalibrationMemory().get_read_access()

object_detector = ObjectDetector()
object_distance_calculator=ObjectDistanceCalculator()
try:
    import time
    cam_mats = camera_calibration_memory.read()
    K = cam_mats[0, :3, :3]
    P = cam_mats[1]
    print("[DEBUG] Camera intrinsics K:\n", K)
    print("[DEBUG] Camera extrinsics P:\n", P)
    while True:
        # Convert to Torch tensor and normalize
        frame=rgb_camera_memory.read()
        depth_map = depth_camera_memory.read()
        radar_data = radar_memory.read()
        if np.count_nonzero(frame) == 0:
            # No data yet, skip this iteration  
            continue
        # Detect objects
        boxes, class_ids, scores =object_detector.get_objects(frame)
        # Get distance for each object
        #distances=object_distance_calculator.get_depth_camera_distances(boxes,depth_map)

        # Debugging step
        distances = object_distance_calculator.get_radar_distances(boxes, radar_data, K, P, constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT, mode="range")

        # Visualise
        visualiser= POVVisualiser(
            object_detector.classes,
            frame,boxes,
            class_ids,
            scores,
            distances)

        visualiser.overlay_radar_points(
            radar_points_world=radar_data,
            K=K,
            P=P,
            img_w=constants.IMAGE_WIDTH,
            img_h=constants.IMAGE_HEIGHT,
            point_radius=2,
            color_mode='depth'  # or 'fixed'
        )
        visualiser.show()

finally:
    cv2.destroyAllWindows()


