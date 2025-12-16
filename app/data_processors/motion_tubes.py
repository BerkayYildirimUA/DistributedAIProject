import numpy as np
import cv2

class MotionTubeProjector:
    """
    Projects a lane tube through 3D points (ground z=0) with a camera.
    Assumptions:
    - Camera is centered in the car, height cam_height above the road surface.
    - Road surface will be flat (mostly).
    - OpenCV camera frame: Z = forward, X = right, Y = down (towards the bottom of the image).
    - We generate the path in the vehicle frame and map it to the camera frame.
    """

    def __init__(self,
                 img_w: int,                            # resolution of the camera (w x h)
                 img_h: int,
                 fov_deg: float = 90.0,
                 cam_height: float = 1.5,               # height of cam above the ground
                 lane_width: float = 3.6,
                 wheelbase: float = 2.8,
                 max_steer_rad: float = np.deg2rad(55),
                 meters_ahead: float = 40.0,            # maximum horizon I want to draw
                 ema_beta: float = 0.85,                # parameter for smoothing the steering angle
                 center_offset_m: float = 0.0):
        self.img_w, self.img_h = img_w, img_h
        self.cam_h = float(cam_height)
        self.lane_w = float(lane_width)
        self.wb = float(wheelbase)
        self.max_steer = float(max_steer_rad)
        self.meters_ahead = float(meters_ahead)
        self.center_offset = float(center_offset_m)

        # intrinsics from FOV (fx=fy for square pixels)
        f = 0.5 * img_w / np.tan(np.deg2rad(fov_deg) / 2.0)
        self.K = np.array([[f, 0, img_w / 2.0],
                           [0, f, img_h / 2.0],
                           [0, 0, 1.0]], dtype=np.float32)
        self.dist = np.zeros(5, dtype=np.float32)  # no distortion

        # smoothing of steer angle
        self.ema_beta = float(ema_beta)
        self._steer_filt = 0.0

    def _smooth_steer(self, steer_rad: float) -> float:
        s = float(np.clip(steer_rad, -self.max_steer, self.max_steer))
        self._steer_filt = self.ema_beta * self._steer_filt + (1.0 - self.ema_beta) * s
        return self._steer_filt

    def _centerline_xy(self, speed_ms: float, steer_rad: float) -> np.ndarray:
        """
        Generate a 2D path in vehicle coordinates:
        x = forward (m), y = left+ (m). Length scaling with speed.
        """
        steer = self._smooth_steer(steer_rad)
        # we can draw horizon longer when speed is higher
        horizon_m = float(np.clip(speed_ms * 2.0, 30.0, self.meters_ahead))
        s = np.linspace(0.0, horizon_m, int(horizon_m * 4) + 2)  # ~4 samples/m

        kappa = np.tan(steer) / max(self.wb, 1e-6)
        if abs(kappa) < 1e-6:
            x = s
            y = np.zeros_like(s)
        else:
            x = np.sin(kappa * s) / kappa
            y = (1.0 - np.cos(kappa * s)) / kappa

        # lateral offset of the ego-lane center (fine tuning if necessary, standard offset on 0)
        y = y + self.center_offset
        return np.stack([x, y], axis=1)  # [N,2], x fwd, y left

    def _veh_to_cam_points(self, xy: np.ndarray, side_offset: float) -> np.ndarray:
        """
        Put (x fwd, y left) -> camera frame (X right, Y down, Z fwd) on ground.
        side_offset = +half lane (left) or -half lane (right)
        """
        x = xy[:, 0]
        y_left = xy[:, 1] + side_offset               # to left positive
        X_right = -y_left                             # in cam: right is positive
        Y_down  = np.full_like(x, self.cam_h)         # ground is cam_height under camera
        Z_fwd   = x                                   # forward
        pts = np.stack([X_right, Y_down, Z_fwd], axis=1).astype(np.float32)
        # remove points behind the camera or those that are very close
        return pts[Z_fwd > 0.5]

    # the function _project is used to project to pixels
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

    def get_projected_lanes(self,speed_ms: float, steer_rad: float):
        center_xy = self._centerline_xy(speed_ms, steer_rad)
        half = 0.5 * self.lane_w

        left_pts_cam  = self._veh_to_cam_points(center_xy, +half)
        right_pts_cam = self._veh_to_cam_points(center_xy, -half)
        center_pts_cam = self._veh_to_cam_points(center_xy, 0)

        left_uv  = self._project(left_pts_cam)
        right_uv = self._project(right_pts_cam)
        center_uv = self._project(center_pts_cam)
        return self._to_poly(left_uv),self._to_poly(center_uv) ,self._to_poly(right_uv)

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


