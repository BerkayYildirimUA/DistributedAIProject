import numpy as np
import cv2

class MotionTubeProjector:
    """
    Projecteert een lane-tube door 3D punten (grond z=0) met pinhole camera.
    Assumpties:
    - Camera staat in het midden van de auto, hoogte cam_height boven wegdek.
    - OpenCV camera-frame: Z = voorwaarts, X = rechts, Y = omlaag (naar beeldbodem).
    - We genereren het pad in het voertuigframe en mappen naar camera-frame.
    """

    def __init__(self,
                 img_w: int,
                 img_h: int,
                 fov_deg: float = 90.0,
                 cam_height: float = 1.5,
                 lane_width: float = 3.6,
                 wheelbase: float = 2.8,
                 max_steer_rad: float = np.deg2rad(35),
                 meters_ahead: float = 40.0,
                 ema_beta: float = 0.85,
                 center_offset_m: float = 0.0):
        self.img_w, self.img_h = img_w, img_h
        self.cam_h = float(cam_height)
        self.lane_w = float(lane_width)
        self.wb = float(wheelbase)
        self.max_steer = float(max_steer_rad)
        self.meters_ahead = float(meters_ahead)
        self.center_offset = float(center_offset_m)

        # intrinsics uit FOV (fx=fy voor square pixels)
        f = 0.5 * img_w / np.tan(np.deg2rad(fov_deg) / 2.0)
        self.K = np.array([[f, 0, img_w / 2.0],
                           [0, f, img_h / 2.0],
                           [0, 0, 1.0]], dtype=np.float32)
        self.dist = np.zeros(5, dtype=np.float32)  # geen distortion

        # smoothing van stuurhoek
        self.ema_beta = float(ema_beta)
        self._steer_filt = 0.0

    def _smooth_steer(self, steer_rad: float) -> float:
        s = float(np.clip(steer_rad, -self.max_steer, self.max_steer))
        self._steer_filt = self.ema_beta * self._steer_filt + (1.0 - self.ema_beta) * s
        return self._steer_filt

    def _centerline_xy(self, speed_ms: float, steer_rad: float) -> np.ndarray:
        """
        Genereer een 2D pad in voertuig-frame:
        x = voorwaarts (m), y = links+ (m). Lengte schalen met snelheid.
        """
        steer = self._smooth_steer(steer_rad)
        # horizon langer bij hogere snelheid
        horizon_m = float(np.clip(speed_ms * 2.0, 12.0, self.meters_ahead))
        s = np.linspace(0.0, horizon_m, int(horizon_m * 4) + 2)  # ~4 samples/m

        kappa = np.tan(steer) / max(self.wb, 1e-6)
        if abs(kappa) < 1e-6:
            x = s
            y = np.zeros_like(s)
        else:
            x = np.sin(kappa * s) / kappa
            y = (1.0 - np.cos(kappa * s)) / kappa

        # laterale offset van de ego-lane center (fijn afstellen)
        y = y + self.center_offset
        return np.stack([x, y], axis=1)  # [N,2], x fwd, y left

    def _veh_to_cam_points(self, xy: np.ndarray, side_offset: float) -> np.ndarray:
        """
        Zet (x fwd, y left) -> camera frame (X right, Y down, Z fwd) op grond.
        side_offset = +half lane (links) of -half lane (rechts)
        """
        x = xy[:, 0]                                  # fwd
        y_left = xy[:, 1] + side_offset               # naar links positief
        X_right = -y_left                             # rechts positief
        Y_down  = np.full_like(x, self.cam_h)         # grond is cam_height onder camera
        Z_fwd   = x                                   # voorwaarts
        pts = np.stack([X_right, Y_down, Z_fwd], axis=1).astype(np.float32)
        # verwijder punten achter de camera of heel dichtbij
        return pts[Z_fwd > 0.5]

    def _project(self, pts_cam: np.ndarray) -> np.ndarray:
        if pts_cam.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32)
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)
        img_pts, _ = cv2.projectPoints(pts_cam, rvec, tvec, self.K, self.dist)
        uv = img_pts.reshape(-1, 2)
        # in-bounds filter
        m = (uv[:, 0] >= 0) & (uv[:, 0] < self.img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < self.img_h)
        return uv[m]

    @staticmethod
    def _to_poly(uv: np.ndarray) -> np.ndarray:
        return np.round(uv).astype(np.int32).reshape((-1, 1, 2))

    def project_and_draw(self, frame_bgr: np.ndarray, speed_ms: float, steer_rad: float,
                         color=(0, 255, 255), thickness: int = 4) -> np.ndarray:
        center_xy = self._centerline_xy(speed_ms, steer_rad)
        half = 0.5 * self.lane_w

        left_pts_cam  = self._veh_to_cam_points(center_xy, +half)
        right_pts_cam = self._veh_to_cam_points(center_xy, -half)

        left_uv  = self._project(left_pts_cam)
        right_uv = self._project(right_pts_cam)

        if left_uv.shape[0] > 1:
            cv2.polylines(frame_bgr, [self._to_poly(left_uv)],  False, color, thickness, cv2.LINE_AA)
        if right_uv.shape[0] > 1:
            cv2.polylines(frame_bgr, [self._to_poly(right_uv)], False, color, thickness, cv2.LINE_AA)
        return frame_bgr

    def compute_tube_points_img(self, speed_ms: float, steer_rad: float):
        """
        Geeft (left_uv, right_uv) terug als float32 Nx2 in beeldpixels (u,v).
        Sluit aan op project_and_draw (zelfde logica, maar zonder tekenen).
        """
        center_xy = self._centerline_xy(speed_ms, steer_rad)
        half = 0.5 * self.lane_w

        left_pts_cam = self._veh_to_cam_points(center_xy, +half)
        right_pts_cam = self._veh_to_cam_points(center_xy, -half)

        left_uv = self._project(left_pts_cam)
        right_uv = self._project(right_pts_cam)
        return left_uv.astype(np.float32), right_uv.astype(np.float32)


