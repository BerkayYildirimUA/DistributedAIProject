import cv2
import numpy as np
import constants

from data_processors.object_detector import ObjectDetector
from data_processors.object_distance_calculator import ObjectDistanceCalculator
from memory.shared_memory import RGBCameraMemory, DepthCameraMemory, VehicleDistanceMemory, RadarMemory, CameraCalibrationMemory
from engine.pov_visualiser import POVVisualiser
from data_processors.radar_points_projector import RadarPointsProjector

# Attach to shared memory
rgb_camera_memory = RGBCameraMemory().get_read_access()
depth_camera_memory = DepthCameraMemory().get_read_access()
#vehicle_distance_memory = VehicleDistanceMemory().get_write_access()
radar_memory = RadarMemory().get_read_access()
camera_calibration_memory = CameraCalibrationMemory().get_read_access()

object_detector = ObjectDetector()
object_distance_calculator=ObjectDistanceCalculator()
radar_points_projector = RadarPointsProjector()
try:
    import time
    cam_mats = camera_calibration_memory.read()
    K = np.asarray(cam_mats[0, :3, :3], dtype=np.float64)
    P = np.asarray(cam_mats[1], dtype=np.float64)
    R = P[:3, :3]

    # rotation should be orthonormal and proper
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-5), "P[:3,:3] not orthonormal"
    detR = np.linalg.det(R)
    assert -0.9 < detR < -1.1, f"det(R) ~ {detR}, expected +1"

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

        # projection of radar_points to camera
        u, v, z, Pc, kept = RadarPointsProjector.project_radar_points_world_to_image(
            radar_data, K, P, constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT
        )

        distances = object_distance_calculator.get_radar_distances(
            boxes,
            box_pad=2.0,
            robust_pct=0.3,
            mode="range",  # uses np.linalg.norm(Pc, axis=1)
            projection=(u, v, z, Pc, kept)
        )

        # Visualise
        visualiser= POVVisualiser(
            object_detector.classes,
            frame,boxes,
            class_ids,
            scores,
            distances)

        visualiser.overlay_radar_points(
            point_radius=2,
            color_mode='depth',
            projection=(u, v, z, Pc, kept)
        )
        visualiser.show()

finally:
    cv2.destroyAllWindows()


