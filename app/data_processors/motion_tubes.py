import numpy as np

class MotionTubeProjector:
    """
    Draws "motion tubes" (left/center/right lane boundaries) by:
    - creating a simple path in the vehicle frame (x = how many m forward, y how many m left)
    - shifting it by lane width
    - projecting to image pixels with a pinhole camera model

    Assumptions:
    - Flat ground
    - Camera is cam_height meters above the ground
    - Camera frame: X right, Y down, Z forward
    - No lens distortion
    """

    def __init__(
        self,
        img_w:      int,
        img_h:      int,
        fov_deg:    float = 90.0,
        cam_height: float = 1.5,
        lane_width: float = 3.6,
        wheelbase:  float = 2.8,
        max_steer_rad:  float = np.deg2rad(55),
        meters_ahead:   float = 40.0,               # max I want to draw
        ema_beta:   float = 0.85,                   # smoothing factor for steering
        center_offset_m: float = 0.0,               # offset if the tube is not nice in the middle, you can tune this
    ):
        self.w = int(img_w)
        self.h = int(img_h)

        self.cam_h          = float(cam_height)
        self.lane_w         = float(lane_width)
        self.wb             = float(wheelbase)
        self.max_steer      = float(max_steer_rad)
        self.meters_ahead   = float(meters_ahead)
        self.center_offset  = float(center_offset_m)

        # Simple intrinsics from HFOV (fy derived from aspect ratio)
        hfov = np.deg2rad(float(fov_deg))
        self.fx = self.w / (2.0 * np.tan(hfov / 2.0))
        vfov = 2.0 * np.arctan((self.h / self.w) * np.tan(hfov / 2.0))
        self.fy = self.h / (2.0 * np.tan(vfov / 2.0))
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0

        # steering smoothing (EMA)
        self.beta = float(ema_beta)
        self._steer_filt = 0.0

    def _smooth_steer(self, steer_rad: float) -> float:
        steer = float(np.clip(steer_rad, -self.max_steer, self.max_steer))
        self._steer_filt = self.beta * self._steer_filt + (1.0 - self.beta) * steer
        return self._steer_filt

    def _make_centerline(self, speed_ms: float, steer_rad: float) -> np.ndarray:
        """
        Returns Nx2 array: [x_forward, y_left]
        """
        steer = self._smooth_steer(steer_rad)

        # horizon: longer when faster (same idea as your original code)
        horizon = float(np.clip(speed_ms * 2.0, 30.0, self.meters_ahead))
        s = np.linspace(0.0, horizon, int(horizon * 4) + 2)  # ~4 samples per meter

        kappa = np.tan(steer) / self.wb  # curvature
        if abs(kappa) < 1e-6:
            x = s
            y = np.zeros_like(s)
        else:
            x = np.sin(kappa * s) / kappa
            y = (1.0 - np.cos(kappa * s)) / kappa

        y = y + self.center_offset
        return np.stack([x, y], axis=1)

    def _project_lane(self, center_xy: np.ndarray, side_offset: float) -> np.ndarray:
        """
        Converts centerline -> lane boundary -> image pixels.
        Returns Nx2 float pixels (u,v).
        """
        x = center_xy[:, 0]                       # forward
        y = center_xy[:, 1] + side_offset         # left (+)

        # Vehicle frame -> camera frame (X right, Y down, Z forward)
        X = -y                                    # right is positive
        Y = np.full_like(x, self.cam_h)           # ground is cam_h below camera
        Z = x

        # ignore points too close/behind camera
        m = Z > 0.5
        X, Y, Z = X[m], Y[m], Z[m]
        if Z.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        # pinhole projection
        u = self.fx * (X / Z) + self.cx
        v = self.fy * (Y / Z) + self.cy

        # keep points that land inside the image
        inside = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        uv = np.stack([u[inside], v[inside]], axis=1).astype(np.float32)
        return uv

    def get_projected_lanes(self, speed_ms: float, steer_rad: float):
        """
        Returns (left_poly, center_poly, right_poly) as int arrays shaped for cv2.polylines:
        each is (N, 1, 2) with int pixel coordinates.
        """
        center_xy = self._make_centerline(speed_ms, steer_rad)
        half = 0.5 * self.lane_w

        left_uv   = self._project_lane(center_xy, +half)
        center_uv = self._project_lane(center_xy, 0.0)
        right_uv  = self._project_lane(center_xy, -half)

        def to_poly(uv: np.ndarray) -> np.ndarray:
            return np.round(uv).astype(np.int32).reshape((-1, 1, 2))

        return to_poly(left_uv), to_poly(center_uv), to_poly(right_uv)
