import cv2
import numpy as np

from data_processors.object_detector import ObjectDetector
from data_processors.object_distance_calculator import ObjectDistanceCalculator
from memory.shared_memory import RGBCameraMemory, DepthCameraMemory, VehicleDistanceMemory, RadarMemory
from engine.pov_visualiser import POVVisualiser

# Attach to shared memory
rgb_camera_memory = RGBCameraMemory().get_read_access()
depth_camera_memory = DepthCameraMemory().get_read_access()
vehicle_distance_memory = VehicleDistanceMemory().get_write_access()
radar_memory = RadarMemory().get_read_access()

object_detector = ObjectDetector()
object_distance_calculator=ObjectDistanceCalculator()
try:
    import time
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
        print(f"Num Radar Detections: {len(radar_data)} with max depth: {np.max(radar_data[:, 0]) if len(radar_data)>0 else 'N/A'}")
        distances = object_distance_calculator.get_radar_distances(boxes, radar_data)

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


