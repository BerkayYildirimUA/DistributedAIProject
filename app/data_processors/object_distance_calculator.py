import numpy as np
from sklearn.cluster import DBSCAN


class ObjectDistanceCalculator:

    def get_depth_camera_distances(self, object_boxes, depth_map=None):
        distance=[]
        for (x1, y1, x2, y2) in object_boxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if depth_map is not None:
                # clip coords to image dimensions
                x1d = max(0, min(depth_map.shape[1] - 1, x1))
                x2d = max(0, min(depth_map.shape[1] - 1, x2))
                y1d = max(0, min(depth_map.shape[0] - 1, y1))
                y2d = max(0, min(depth_map.shape[0] - 1, y2))
                crop = depth_map[y1d:y2d + 1, x1d:x2d + 1]
                if crop.size > 0:
                    distance.append(np.nanmin(crop))
                else:
                    raise Exception("No distance could be calculated!")
        if len(distance) != len(object_boxes):
            raise Exception("Object distance calculation failed: size mismatch between distances and found object boxes!")
        return distance

    def get_radar_distances(self, object_boxes, projection, box_pad=2.0, robust_pct=0.3, mode='z'):
        boxes = np.asarray(object_boxes, dtype=np.float64).reshape(-1, 4)
        u, v, z, Pc, kept = projection

        if u.size == 0:
            return [float('nan')] * len(boxes)

        # Precompute Euclidean range in camera frame if needed
        ranges = np.linalg.norm(Pc, axis=1) if mode == 'range' else None

        distances = []
        for (x1, y1, x2, y2) in boxes:
            # small padding to absorb tiny calibration/timing errors
            x1p = x1 - box_pad
            y1p = y1 - box_pad
            x2p = x2 + box_pad
            y2p = y2 + box_pad

            m = (u >= x1p) & (u <= x2p) & (v >= y1p) & (v <= y2p)
            if not np.any(m):
                distances.append(float('nan'))
                continue

            arr = ranges[m] if mode == 'range' else z[m]
            k = max(1, int(robust_pct * arr.size))
            idx = np.argpartition(arr, k - 1)[:k]
            distances.append(float(np.median(arr[idx])))

        return distances







