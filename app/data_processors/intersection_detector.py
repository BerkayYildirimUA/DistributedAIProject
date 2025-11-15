import numpy as np
import cv2

class IntersectionDetector:
    def __init__(self):
        pass

    def is_intersecting_list(self, lane_a, lane_b, boxes):
        if len(lane_a) < 2 or len(lane_b) < 2:
            return [False] * len(boxes)

            # Convert to numpy arrays and ensure shape (N,2)
        lane_a = np.array(lane_a, dtype=np.int32).reshape(-1, 2)
        lane_b = np.array(lane_b, dtype=np.int32).reshape(-1, 2)

        # Form a closed polygon by connecting lane_a and reversed lane_b
        lane_poly = np.vstack([lane_a, lane_b[::-1]]).astype(np.int32)

        results = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Use pointPolygonTest
            inside = cv2.pointPolygonTest(lane_poly, (cx, cy), False) >= 0
            results.append(inside)

        return results
