import numpy as np
import constants
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import numpy as np


class ObjectDistanceCalculator:
    def __init__(self):
        self.last_valid_distances = {}
        
        # --- 1. Calculate Camera Intrinsics (K) ---
        HFOV_RAD = constants.HOR_FOV_RAD
        VFOV_RAD = constants.VERT_FOV_RAD
        
        self.fx = constants.IMAGE_WIDTH / (2.0 * np.tan(HFOV_RAD / 2.0)) # focal length in pixels x
        self.fy = constants.IMAGE_HEIGHT / (2.0 * np.tan(VFOV_RAD / 2.0)) # focal length in pixels y
        
        self.cx = constants.IMAGE_WIDTH / 2.0 # optical center x
        self.cy = constants.IMAGE_HEIGHT / 2.0 # optical center y
        
        # The 3x3 Intrinsic Matrix K
        # Encodes how 3D points map to pixel coordinates
        self.K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ])

        # --- 2. Filtering Constants ---
        # Used to remove ego-vehicle reflection (0.0m) and distant background
        self.MIN_VALID_DEPTH = 3.0   
        self.MAX_TARGET_DEPTH = 100.0 


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
        CENTER_RATIO = 0.35
        distances = []

        if len(radar_data) == 0 or len(object_boxes) == 0:
            return [np.nan] * len(object_boxes)

        # --- Convert radar data to Cartesian coordinates ---
        radar_data = np.array(radar_data)
        depths = radar_data[:, 0]
        azimuths = radar_data[:, 2]
        altitudes = radar_data[:, 3]

        cos_alt = np.cos(altitudes)
        X = depths * cos_alt * np.cos(azimuths)
        Y = depths * cos_alt * np.sin(azimuths)
        Z = depths * np.sin(altitudes)

        # --- Convert to camera frame ---
        # Bring 3D radar coordinates into the same coordinate convention as the camera
        P_cam = np.stack([Y, -Z, X], axis=1)

        # Only keep points in front of camera
        # third dimension (zero indexed) represents depth, should be greater than 0
        valid_mask = P_cam[:, 2] > 0
        P_cam = P_cam[valid_mask]
        depths = depths[valid_mask]

        # --- Project to image plane ---
        # Pinhole camera model: maps 3D camera coordinates to 2D image pixels
        pixels_h = (self.K @ P_cam.T).T  # matrix multiplication

        # converts homogeneous coordinates to true pixel coordinates
        u = pixels_h[:, 0] / pixels_h[:, 2]
        v = pixels_h[:, 1] / pixels_h[:, 2]

        # Keep only pixels within frame
        # Some projected radar points might fall outside the camera's visible area
        mask = (
                (u >= 0) & (u < constants.IMAGE_WIDTH) &
                (v >= 0) & (v < constants.IMAGE_HEIGHT)
        )
        u, v, depths = u[mask], v[mask], depths[mask]

        # --- Build KD-tree for fast spatial lookup ---
        # creates a spatial index over all projected radar pixels with coordinates (u, v)
        pixel_tree = cKDTree(np.stack([u, v], axis=1))

        # --- For each bounding box, query nearby radar points ---
        for i, (x1, y1, x2, y2) in enumerate(object_boxes):
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2, y1 + h / 2

            x1_inner = cx - (w * CENTER_RATIO / 2)
            x2_inner = cx + (w * CENTER_RATIO / 2)
            y1_inner = cy - (h * CENTER_RATIO / 2)
            y2_inner = cy + (h * CENTER_RATIO / 2)

            # Find points roughly inside box (fast KD-tree range query)
            box_center = np.array([(x1_inner + x2_inner) / 2, (y1_inner + y2_inner) / 2])
            box_radius = max((x2_inner - x1_inner), (y2_inner - y1_inner)) / 2
            idxs = pixel_tree.query_ball_point(box_center, box_radius)

            if not idxs:
                distances.append(self.last_valid_distances.get(i, np.nan))
                continue

            d_in_box = depths[idxs]
            d_in_box = d_in_box[
                (d_in_box > self.MIN_VALID_DEPTH) &
                (d_in_box < self.MAX_TARGET_DEPTH)
                ]
            if len(d_in_box) == 0:
                distances.append(self.last_valid_distances.get(i, np.nan))
                continue

            # --- Clustering ---
            # use DBSCAN to cluster the distances. Some distances will be part of the detected vehicle, some
            # distances will be part of distant surfaces that are coincidentally within the bounding box.
            if len(d_in_box) > 6:
                db = DBSCAN(eps=0.5, min_samples=2).fit(d_in_box.reshape(-1, 1))  # allow 0.5m distance between points
                labels = db.labels_
                valid_clusters = [d_in_box[labels == l] for l in set(labels) if l != -1]
                if valid_clusters:
                    # TODO: test both options, see which one performs best
                    # First option
                    # cluster_means = [np.mean(c) for c in valid_clusters]
                    # val = cluster_means[np.argmin(cluster_means)]

                    # Second option
                    largest_cluster = max(valid_clusters, key=len)
                    val = np.median(largest_cluster)
                else:
                    val = np.median(d_in_box)

                self.last_valid_distances[i] = val
                distances.append(val)

        return distances




