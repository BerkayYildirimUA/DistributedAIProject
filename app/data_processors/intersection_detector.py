import numpy as np
import cv2

class IntersectionDetector:
    def __init__(self):
        pass

    def point_to_segment_distance(self,px, py, x1, y1, x2, y2):
        """
        Computes the distance between a point (px,py) and a line segment (x1,y1)-(x2,y2).
        """
        # Handle zero-length segment
        seg_dx = x2 - x1
        seg_dy = y2 - y1
        seg_len_sq = seg_dx ** 2 + seg_dy ** 2
        if seg_len_sq == 0:
            return np.hypot(px - x1, py - y1)

        # Projection factor t of point onto the segment (parametric)
        t = ((px - x1) * seg_dx + (py - y1) * seg_dy) / seg_len_sq
        t = max(0, min(1, t))  # clamp to segment

        # Closest point on segment
        proj_x = x1 + t * seg_dx
        proj_y = y1 + t * seg_dy

        return np.hypot(px - proj_x, py - proj_y)

    def is_intersecting_list_trajectory_based(self, boxes, center_line, distance_to_line, margin):
        results = []

        # Convert center line into list of consecutive segment pairs
        center_line = [tuple(p[0]) for p in center_line]  # convert (1,2) arrays → (x, y) tuples
        segments = list(zip(center_line[:-1], center_line[1:]))
        for box in boxes:
            # Extract object center
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Compute minimum distance to the trajectory
            min_dist = float("inf")
            for (p1, p2) in segments:
                d = self.point_to_segment_distance(cx, cy, p1[0], p1[1], p2[0], p2[1])
                if float(d) < float(min_dist):
                    min_dist = d

            # Check threshold
            threshold = distance_to_line + margin
            # print(float(min_dist),threshold)
            results.append(float(min_dist) <= float(threshold))

        return results



