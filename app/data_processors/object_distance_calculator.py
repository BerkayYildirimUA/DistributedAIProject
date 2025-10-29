import numpy as np

class ObjectDistanceCalculator:
    def __init__(self):
        self.last_valid_distances = {}
    
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

    def get_radar_distances(self, object_boxes, radar_data, image_width, image_height):
        distances = []
    
        # Convert radar detections to pixel-like x positions for approximate mapping
        # No polar to cartesian conversion is needed because we can map the azimuth straight to an x coordinate through normalization

        # Convert azimuth angle in the range of -pi/2 to pi/2 to normalized image coordinate in range of [0, image_width]
        radar_points = []
        for detection in radar_data:
            depth = detection[0]
            azimuth = detection[2]
            hfov = np.deg2rad(90.0) # azimuth of CARLA radar is expressed in radians
            x_norm = (azimuth + hfov / 2) / hfov
            x_pixel = np.clip(x_norm * image_width, 0, image_width - 1)
            radar_points.append((x_pixel, depth))
    
        for i, (x1, y1, x2, y2) in enumerate(object_boxes):
            # Find radar points within the bounding box horizontal range
            in_box_depths = [d for (rx, d) in radar_points if x1 <= rx <= x2]
            if in_box_depths:
                val = np.median(in_box_depths)
                self.last_valid_distances[i] = val
                distances.append(val)
            else:
                # keep last known value if available
                distances.append(self.last_valid_distances.get(i, np.nan))

        return distances

