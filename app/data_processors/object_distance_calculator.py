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

    def get_radar_distances(self, object_boxes, radar_xyz_world, K, T_cam_world, img_w, img_h):

        if radar_xyz_world.size == 0:
            return [float('nan')] * len(object_boxes)

        # 1) World -> camera (Unreal camera frame)
        Pw = np.hstack([radar_xyz_world, np.ones((radar_xyz_world.shape[0], 1))])  # (N,4)
        Pc_unreal_h = (T_cam_world @ Pw.T).T
        Pc_unreal = Pc_unreal_h[:, :3]

        # 2) Unreal camera -> CV camera (x right, y down, z forward)
        R_ue2cv = np.array([[0, 1, 0],
                            [0, 0, -1],
                            [1, 0, 0]], dtype=float)
        Pc = (R_ue2cv @ Pc_unreal.T).T  # (N,3)

        # 3) Keep points in front of camera (z>0 in CV frame)
        z = Pc[:, 2]
        in_front = z > 0
        Pc = Pc[in_front]
        z = z[in_front]
        if Pc.shape[0] == 0:
            return [float('nan')] * len(object_boxes)

        # 4) Project to pixels
        uvw = (K @ Pc.T).T  # (N,3)
        u = uvw[:, 0] / uvw[:, 2]
        v = uvw[:, 1] / uvw[:, 2]

        # 5) Keep points inside the image
        in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        u, v, z = u[in_img], v[in_img], z[in_img]
        if u.size == 0:
            return [float('nan')] * len(object_boxes)

        distances = []
        for (x1, y1, x2, y2) in np.asarray(object_boxes):
            # Optional: pad boxes slightly to be robust to small calibration errors
            pad = 2.0
            x1p, y1p, x2p, y2p = x1 - pad, y1 - pad, x2 + pad, y2 + pad

            mask = (u >= x1p) & (u <= x2p) & (v >= y1p) & (v <= y2p)
            if not np.any(mask):
                distances.append(float('nan'))  # better than 0.0 to mean "no data"
            else:
                z_box = z[mask]
                # robust choice: median of the closest 30% depths (guards against background points)
                k = max(1, int(0.3 * z_box.size))
                idx = np.argpartition(z_box, k - 1)[:k]
                distances.append(float(np.median(z_box[idx])))

        return distances





