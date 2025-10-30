import cv2
import numpy as np

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
bird_eye_visualiser=BirdVisualiser()
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

        # Overlay lanes

        lanes = lane_detector.get_lanes(frame)

        # Visualise
        visualiser= POVVisualiser(
            object_detector.classes,
            frame,boxes,
            class_ids,
            scores,
            distances,
            lanes)
        visualiser.show()

        # Dummy video frames data
        video_frames_data = [(
                boxes,  # boxes
                class_ids,  # class_ids
                lanes[0],  # lane left
                lanes[1]  # lane right
            )]


        vis = BirdVisualiser()
        vis.animate(video_frames_data, interval=100)  # 100ms per frame (~10 FPS)

        # bird_eye_visualiser.show(boxes,class_ids,lanes[0],lanes[1])

finally:
    cv2.destroyAllWindows()


