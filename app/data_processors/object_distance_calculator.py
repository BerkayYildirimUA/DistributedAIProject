import numpy as np
import constants
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
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

    def get_radar_distances(self, object_boxes, radar_data):

        distances = []

        # Project radar points to 2D camera frame
        radar_points_2d = []
        for detection in radar_data:
            depth = detection[0]
            azimuth = detection[2]
            altitude = detection[3]

            # Convert spherical to Cartesian coordinates
            x = depth * np.cos(altitude) * np.cos(azimuth)
            y = depth * np.cos(altitude) * np.sin(azimuth)
            z = depth * np.sin(altitude)

            # Project to 2D using camera intrinsics (assumed available)
            point_3d = np.array([x, y, z, 1.0])
            point_camera = constants.RADAR_TO_CAMERA_TRANSFORM @ point_3d
            point_2d = constants.CAMERA_INTRINSICS @ point_camera[:3]

            u = point_2d[0] / point_2d[2]
            v = point_2d[1] / point_2d[2]

            radar_points_2d.append((u, v, depth))

        # For each bounding box, find radar points inside and cluster their depths
        for (x1, y1, x2, y2) in object_boxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Filter radar points inside the bounding box
            box_points = [depth for (u, v, depth) in radar_points_2d if x1 <= u <= x2 and y1 <= v <= y2]

            if not box_points:
                distances.append(np.nan)
                continue

            # Cluster depths using DBSCAN
            X = np.array(box_points).reshape(-1, 1)
            clustering = DBSCAN(eps=2.0, min_samples=3).fit(X)

            labels = clustering.labels_
            if len(set(labels)) <= 1:
                distances.append(np.mean(box_points))
                continue

            # Find largest cluster
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            largest_cluster_label = unique_labels[np.argmax(counts)]
            cluster_points = X[labels == largest_cluster_label]

            # Use mean depth of largest cluster
            distances.append(float(np.mean(cluster_points)))

        return distances




