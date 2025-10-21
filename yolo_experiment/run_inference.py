import cv2
import numpy as np
from object_detection import ObjectDetection

# Replace with the shared memory name printed by the writer
filename = "shared_frame.dat"
image_shape = (480, 640, 3)
dtype = np.uint8

depth_filename = "shared_depth.dat"
depth_shape = (480, 640)
dtype_depth = np.float32

# Attach to shared memory
shared_frame = np.memmap(filename, dtype=dtype, mode='r', shape=image_shape)
depth_shared = np.memmap(depth_filename, dtype=dtype_depth, mode='r', shape=depth_shape)

# Create shared memory
vehicle_distance_filename = "vehicle_distance.dat"
vehicle_distance_shape = (1, 1)
shared_vehicle_distance = np.memmap(vehicle_distance_filename, dtype=np.float32, mode='w+', shape=vehicle_distance_shape)

object_detection=ObjectDetection()
try:
    import time
    while True:
        # Convert to Torch tensor and normalize
        frame=shared_frame.copy()
        depth_map = depth_shared.copy()
        if np.count_nonzero(frame) == 0:
            # No data yet, skip this iteration
            continue
        frame_with_boxes, distance_vehicle_in_front_m =object_detection.detect_and_add_overlay(frame, depth_map)
        shared_vehicle_distance[0, 0] = distance_vehicle_in_front_m
        shared_vehicle_distance.flush()
        cv2.imshow("Camera", frame_with_boxes)
        cv2.waitKey(1)
        # time.sleep(0.05)
finally:
    cv2.destroyAllWindows()


