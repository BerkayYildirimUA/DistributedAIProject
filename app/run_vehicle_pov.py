import cv2
import numpy as np

from data_processors.intersection_detector import IntersectionDetector
from engine.bird_eye_visualiser import BirdVisualiser
from data_processors.lane_detector import LaneDetector
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
lane_detector=LaneDetector()
# bird_eye_visualiser=BirdVisualiser(640,480)
intersection_detector=IntersectionDetector()
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
        # Get distance for each object
        distances=object_distance_calculator.get_distances(boxes,depth_map)

        # Lanes
        lanes_a,lanes_b = lane_detector.get_lanes(frame,int_degree=1)
        # Intersection with lane
        is_intersected=intersection_detector.is_intersecting_list(lanes_a,lanes_b,boxes)

        # Visualise
        visualiser= POVVisualiser(
            object_detector.classes,
            frame,boxes,
            class_ids,
            scores,
            distances,
            is_intersected,
            [*lanes_a,*lanes_b],)
        visualiser.show()


        # if len(lanes) > 0:
        #     bird_eye_visualiser.show(boxes,class_ids,lanes)

finally:
    cv2.destroyAllWindows()


