import cv2
import numpy as np
from object_detection import ObjectDetection

# Replace with the shared memory name printed by the writer
filename = "shared_frame.dat"
image_shape = (480, 640, 3)
dtype = np.uint8

# Attach to shared memory
shared_frame = np.memmap(filename, dtype=dtype, mode='r', shape=image_shape)

object_detection=ObjectDetection()
try:
    import time
    while True:
        # Convert to Torch tensor and normalize
        frame=shared_frame.copy()
        if np.count_nonzero(frame) == 0:
            # No data yet, skip this iteration
            continue
        frame_with_boxes=object_detection.detect_and_add_overlay(frame)
        cv2.imshow("Camera", frame_with_boxes)
        cv2.waitKey(1)
        # time.sleep(0.05)
finally:
    cv2.destroyAllWindows()


