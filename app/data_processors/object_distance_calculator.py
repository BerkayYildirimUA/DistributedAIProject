import numpy as np
import constants
from sklearn.cluster import DBSCAN
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

    def get_radar_distances(self, object_boxes, radar_data, K, P):
        distances = []
        if radar_data is None or radar_data.size == 0:
            return [0.0 for _ in object_boxes]

        # Prepare homogeneous world coordinates
        points_world = np.hstack([radar_data[:, :3], np.ones((radar_data.shape[0], 1))])  # (N,4)

        # World → camera
        points_cam = (P @ points_world.T).T  # (N,4)

        # Only keep points in front of camera (z > 0)
        in_front = points_cam[:, 2] > 0
        points_cam = points_cam[in_front]

        # Fourth dimension represent depths (can also be calculated using euclidian distance, maybe try later)
        depths = radar_data[in_front, 3]

        # Camera --> image (project to pixels)
        pixels = (K @ points_cam[:, :3].T).T  # (N,3)
        u = pixels[:, 0] / pixels[:, 2]
        v = pixels[:, 1] / pixels[:, 2]

        # Combine projected pixel coordinates and their respective depth
        projected = np.vstack([u, v, depths]).T  # (N,3)

        object_boxes = np.array(object_boxes)
        for i, (x1, y1, x2, y2) in enumerate(object_boxes):
            inside_mask = (projected[:, 0] >= x1) & (projected[:, 0] <= x2) & \
                          (projected[:, 1] >= y1) & (projected[:, 1] <= y2)

            inside_points = projected[inside_mask]

            if inside_points.shape[0] == 0:
                distances.append(0.0)
            else:
                # Option 1: mean depth of all points inside box
                # Option 2: cluster using scikit-learn DBScan and choose mean of largest cluster
                # testing this first
                distances.append(float(np.mean(inside_points[:, 2])))

        return distances




