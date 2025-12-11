import numpy as np


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

    def get_radar_distances(self, object_boxes, projection, radar_data, proj_indices):
        boxes = np.asarray(object_boxes, dtype=np.float64).reshape(-1, 4)
        u, v, z, Pc, kept = projection
        if u.size == 0:
            return [float('nan')] * len(boxes)
        per_point_dist = radar_data[proj_indices, 3].astype(np.float64)  # det.depth
        distances = []
        for (x1, y1, x2, y2) in boxes:
            m = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
            if not np.any(m):
                distances.append(float('nan'))
                continue
            distances.append(float(np.min(per_point_dist[m])))
        return distances







