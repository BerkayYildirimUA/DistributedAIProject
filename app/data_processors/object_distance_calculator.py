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

    def get_radar_distances(self, object_boxes, radar_points_world, K, P, img_w, img_h,
                            box_pad=2.0, robust_pct=0.3, mode='z'):

        boxes = np.asarray(object_boxes, dtype=np.float64).reshape(-1, 4)
        if radar_points_world is None or len(radar_points_world) == 0:
            return [float('nan')] * len(boxes)

        # Ensure Nx3
        rpw = np.asarray(radar_points_world, dtype=np.float64)
        if rpw.shape[1] >= 3:
            rpw = rpw[:, :3]
        else:
            raise ValueError(f"radar_points_world must have at least 3 columns, got {rpw.shape}")

        # Homogeneous world points
        Pw = np.hstack([rpw, np.ones((rpw.shape[0], 1), dtype=np.float64)])  # (N,4)
        assert P.shape == (4, 4) and K.shape == (3, 3)

        # World -> camera (CV frame)
        Pc_h = (P @ Pw.T).T  # (N,4)
        Pc = Pc_h[:, :3]  # (N,3)

        # Keep points in front
        z = Pc[:, 2]
        mask_front = z > 0
        if not np.any(mask_front):
            return [float('nan')] * len(boxes)
        Pc = Pc[mask_front]
        z = z[mask_front]

        # Project to pixels
        uvw = (K @ Pc.T).T  # (M,3)
        u = uvw[:, 0] / uvw[:, 2]
        v = uvw[:, 1] / uvw[:, 2]

        # Keep points that land on the image
        in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        if not np.any(in_img):
            return [float('nan')] * len(boxes)
        u, v, z, Pc = u[in_img], v[in_img], z[in_img], Pc[in_img]

        # Precompute Euclidean range if needed
        ranges = np.linalg.norm(Pc, axis=1) if mode == 'range' else None

        distances = []
        for (x1, y1, x2, y2) in boxes:
            # light padding to tolerate tiny calibration errors
            x1p = x1 - box_pad;
            y1p = y1 - box_pad
            x2p = x2 + box_pad;
            y2p = y2 + box_pad

            m = (u >= x1p) & (u <= x2p) & (v >= y1p) & (v <= y2p)
            if not np.any(m):
                distances.append(float('nan'))
                continue

            # robust: take the closest k points and median them
            if mode == 'range':
                arr = ranges[m]
            else:
                arr = z[m]  # optical depth

            k = max(1, int(robust_pct * arr.size))
            idx = np.argpartition(arr, k - 1)[:k]
            distances.append(float(np.median(arr[idx])))

        return distances






