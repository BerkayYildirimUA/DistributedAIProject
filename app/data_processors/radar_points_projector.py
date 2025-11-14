# radar_points_projector.py
import numpy as np

class RadarPointsProjector:

    @staticmethod
    def project_radar_points_world_to_image(radar_points_world, K, P, img_w, img_h):
        if radar_points_world is None or len(radar_points_world) == 0:
            kept = np.zeros(0, dtype=bool)
            out = (np.empty(0),) * 4 + (kept,)
            return out + (np.array([], dtype=np.int64),)

        pts = np.asarray(radar_points_world, dtype=np.float64)

        # Filter padding
        if pts.shape[1] >= 4:
            valid = pts[:, 3] > 0  # depth > 0
        else:
            valid = np.any(pts[:, :3] != 0, axis=1)
        pts_valid = pts[valid]
        if pts_valid.size == 0:
            kept = np.zeros(valid.shape, dtype=bool)
            out = (np.empty(0),) * 4 + (kept,)
            return out + (np.array([], dtype=np.int64),)

        Pw = np.hstack([pts_valid[:, :3], np.ones((pts_valid.shape[0], 1))])
        assert P.shape == (4, 4) and K.shape == (3, 3)
        Pc_h = (P @ Pw.T).T
        Pc = Pc_h[:, :3]

        z = Pc[:, 2]
        in_front = z > 0
        Pc = Pc[in_front]; z = z[in_front]
        if Pc.size == 0:
            kept = np.zeros(valid.shape, dtype=bool)
            kept[valid] = in_front  # propagate
            out = (np.empty(0),) * 4 + (kept,)
            return out + (np.array([], dtype=np.int64),)

        uvw = (K @ Pc.T).T
        u = uvw[:, 0] / uvw[:, 2]
        v = uvw[:, 1] / uvw[:, 2]

        in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        u = u[in_img]; v = v[in_img]; z = z[in_img]; Pc = Pc[in_img]

        kept = np.zeros(valid.shape, dtype=bool)
        kept_indices = np.array([], dtype=np.int64)
        if np.any(in_img):
            kept_indices = np.flatnonzero(valid)[in_front][in_img]
            kept[kept_indices] = True

        out = (u, v, z, Pc, kept)
        return out + (kept_indices,)
