import numpy as np
import constants

class ObjectDistanceCalculator:
    def __init__(self):
        self.last_valid_distances = {}
        self.HOR_FOV_RAD = np.deg2rad(constants.HOR_FOV_DEG)
        self.ASPECT_RATIO = constants.IMAGE_HEIGHT / constants.IMAGE_WIDTH  
        self.CAM_VERT_FOV_RAD = 2 * np.arctan(ASPECT_RATIO * np.tan(HOR_FOV_RAD / 2))
    
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
        
        for detection in radar_data:
            depth = detection[0] # Distance (meters)
            azimuth = detection[2] # Horizontal angle (radians)
            altitude = detection[3] # Vertical angle (radians)

            # Horizontal mapping: azimuth to x_pixel
            # Azimuth is in [-HFOV/2, HFOV/2]. We map it to [0, 1] normalized space.
            x_norm = (azimuth + HFOV_RAD / 2) / HFOV_RAD
            x_pixel = np.clip(x_norm * image_width, 0, image_width - 1)

            # Vertical mapping: altitude to y_pixel
            # Altitude is in [-VFOV_cam/2, VFOV_cam/2].
            # We map it to [0, 1] normalized space (top-to-bottom for positive y).
            # CARLA images have y=0 at the top, y=image_height at the bottom.
            # Positive altitude is up. We need to invert the mapping for the image.
        
            # y_norm = (altitude + VFOV_RAD / 2) / VFOV_RAD # This would map bottom-up (test later if this is actually the case)
            # Corrected for image: [VFOV/2] (top) maps to [0], [-VFOV/2] (bottom) maps to [1]
            y_norm = 1.0 - ((altitude + VFOV_RAD / 2) / VFOV_RAD)
            y_pixel = np.clip(y_norm * image_height, 0, image_height - 1)

            # Store the depth and the 2D projected pixel coordinates
            radar_points_with_pixels.append((x_pixel, y_pixel, depth))
    
        # Filter Radar Points within Bounding Boxes
        for i, (x1, y1, x2, y2) in enumerate(object_boxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
            # Find radar points within the 2D bounding box
            in_box_depths = [
                d for (rx, ry, d) in radar_points_with_pixels 
                if x1 <= rx <= x2 and y1 <= ry <= y2 # 2D Check
            ]
        
            if in_box_depths:
                # Use the median depth of points within box
                val = np.median(in_box_depths)
                self.last_valid_distances[i] = val
                distances.append(val)
            else:
                # Keep last known value if available
                distances.append(self.last_valid_distances.get(i, np.nan))

        return distances

