import numpy as np
import constants

class ObjectDistanceCalculator:
    def __init__(self):
        self.last_valid_distances = {}
        
        # --- 1. Calculate Camera Intrinsics (K) ---
        HFOV_RAD = constants.HOR_FOV_RAD
        VFOV_RAD = constants.VERT_FOV_RAD
        
        self.fx = constants.IMAGE_WIDTH / (2.0 * np.tan(HFOV_RAD / 2.0))
        self.fy = constants.IMAGE_HEIGHT / (2.0 * np.tan(VFOV_RAD / 2.0)) 
        
        self.cx = constants.IMAGE_WIDTH / 2.0
        self.cy = constants.IMAGE_HEIGHT / 2.0
        
        # The 3x3 Intrinsic Matrix K
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
        distances = []
        radar_points_with_pixels = []
        CENTER_RATIO = 0.2

        for detection in radar_data:
            depth = detection[0]
            azimuth = detection[2]
            altitude = detection[3]
            
            # --- 3D Cartesian Conversion (Spherical to Ego-Vehicle Frame) ---
            cos_alt = np.cos(altitude)
            X = depth * cos_alt * np.cos(azimuth)  # Forward
            Y = depth * cos_alt * np.sin(azimuth)  # Right
            Z = depth * np.sin(altitude)           # Up
            
            # --- 3D to 2D Projection (Pinhole Model) ---
            # Map Ego (X=Fwd, Y=Right, Z=Up) to Camera (X'=Right, Y'=Down, Z'=Fwd)
            P_cam_ready = np.array([Y, -Z, X]) 
            
            # Skip points behind the sensor (Z' component must be positive)
            if P_cam_ready[2] <= 0:
                continue 

            P_pixels_homogenous = self.K @ P_cam_ready.T

            # Normalize by the Z component (Depth)
            Z_cam = P_pixels_homogenous[2]
            u = P_pixels_homogenous[0] / Z_cam 
            v = P_pixels_homogenous[1] / Z_cam 

            x_pixel = np.clip(u, 0, constants.IMAGE_WIDTH - 1)
            y_pixel = np.clip(v, 0, constants.IMAGE_HEIGHT - 1)

            radar_points_with_pixels.append((x_pixel, y_pixel, depth))
        
        # Filter Radar Points within Bounding Boxes
        for i, (x1, y1, x2, y2) in enumerate(object_boxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 1. Calculate the dimensions of the central region
            w = x2 - x1
            h = y2 - y1

            x_center = x1 + w / 2
            y_center = y1 + h / 2

            # Define the inner bounding box (60% width and height)
            x1_inner = int(x_center - (w * CENTER_RATIO / 2))
            x2_inner = int(x_center + (w * CENTER_RATIO / 2))
            y1_inner = int(y_center - (h * CENTER_RATIO / 2))
            x2_inner = int(y_center + (h * CENTER_RATIO / 2))  # NOTE: Typo fix here: should be y2_inner

            # Corrected inner y2 coordinate definition
            y2_inner = int(y_center + (h * CENTER_RATIO / 2))

            # 2. Filter points using the inner box and depth filters
            in_box_depths = [
                d for (rx, ry, d) in radar_points_with_pixels
                if x1_inner <= rx <= x2_inner and y1_inner <= ry <= y2_inner
                   and d > self.MIN_VALID_DEPTH and d < self.MAX_TARGET_DEPTH
            ]
            
            if in_box_depths:
                # Use the minimum depth for robustness against volumetric clutter
                val = np.nanmin(in_box_depths)
                self.last_valid_distances[i] = val
                distances.append(val)
            else:
                distances.append(self.last_valid_distances.get(i, np.nan))

        return distances