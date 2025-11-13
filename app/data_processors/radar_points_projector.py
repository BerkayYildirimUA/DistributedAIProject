import numpy as np

class RadarPointsProjector:
    def project_radar_points_world_to_image(self, radar_points_world, K, P, img_w, img_h):
        if radar_points_world is None or len(radar_points_world) == 0:
            return (np.empty(0),) * 4 + (np.zeros(0, dtype=bool),)

        pts = np.asarray(radar_points_world, dtype=np.float64)

        # Filter padded rows
        if pts.shape[1] >= 4:
            valid = pts[:, 3] > 0
        else:
            valid = np.any(pts[:, :3] != 0, axis=1)
        pts = pts[valid]
        if pts.size == 0:
            return (np.empty(0),) * 4 + (valid,)

        # Homogeneous world points
        Pw = np.hstack([pts[:, :3], np.ones((pts.shape[0], 1))])  # (N,4)

        # World -> camera (CV frame)
        assert P.shape == (4, 4) and K.shape == (3, 3)
        Pc_h = (P @ Pw.T).T  # (N,4)
        Pc = Pc_h[:, :3]  # (N,3)

        # Keep points in front (z>0 in CV frame)
        z = Pc[:, 2]
        in_front = z > 0
        Pc = Pc[in_front];
        z = z[in_front]
        if Pc.size == 0:
            kept = np.zeros(valid.shape, dtype=bool)
            kept[valid] = in_front
            return (np.empty(0),) * 4 + (kept,)

        # Project to pixels
        uvw = (K @ Pc.T).T
        u = uvw[:, 0] / uvw[:, 2]
        v = uvw[:, 1] / uvw[:, 2]

        # Keep points that land on the image
        in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        u = u[in_img];
        v = v[in_img];
        z = z[in_img];
        Pc = Pc[in_img]

        # Compose the overall kept mask back to original N
        kept = np.zeros(valid.shape, dtype=bool)
        if np.any(in_img):
            kept_indices = np.flatnonzero(valid)[in_front][in_img]
            kept[kept_indices] = True

        return u, v, z, Pc, kept
