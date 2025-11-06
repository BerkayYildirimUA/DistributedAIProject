import numpy as np
import cv2

class IntersectionDetector:
    def __init__(self):
        pass

    def is_intersecting_list(self, lane_a, lane_b, boxes):
        if len(lane_a) < 2 or len(lane_b) < 2:
            raise ValueError("Not enough lane points provided")

        # Form a closed polygon for the lane region
        lane_poly = np.array(lane_a + lane_b[::-1], dtype=np.int32)

        results = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Check if the center is inside or on the edge of the lane polygon
            inside = cv2.pointPolygonTest(lane_poly, (cx, cy), False) >= 0
            results.append(inside)

        return results
