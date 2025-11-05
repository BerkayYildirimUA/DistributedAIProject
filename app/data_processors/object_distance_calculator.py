import numpy as np
import constants
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import numpy as np


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

    def get_radar_distances(self, object_boxes, radar_data):
        # CARLA radar lives in its own coordinate frame, in order for object_boxes that exist within the
        # camera coordinate frame, and the radar_data that lives in the radar coordinate frame to work together
        # we need to use extrinsic and intrinsic calibrations.
        #
        # Camera intrinsics: Defines how 3D points in the camera coordinate frame are projected into 2D pixels
        # Extrinsics: defines the transform between radar and camera frames
        # --> These are defined in the constants

        distances = []
        radar_np = np.array(radar_data)

        # According to how raw_data is stored in the shared memory
        depth = radar_np[:, 0]
        azimuth = radar_np[:, 2]
        altitude = radar_np[:, 3]

        # Cartesian coordinates in the radar frame
        x_r = depth * np.cos(altitude) * np.cos(azimuth)
        y_r = depth * np.cos(altitude) * np.sin(azimuth)
        z_r = depth * np.sin(altitude)

        # Convert radar 3D points to homogeneous coordinates
        points_radar = np.vstack((x_r, y_r, z_r, np.ones_like(x_r)))

        # Transform from radar to camera frame
        points_cam = constants.CAM_EXTRINSIC @ points_radar

        # Project camera points into 2D image coordinates
        X_c = points_cam[0, :]
        Y_c = points_cam[1, :]
        Z_c = points_cam[2, :]

        # Calculate pixel coordinates in the camera image for each valid radar point
        points_2d_hom = constants.K @ np.vstack((X_c / Z_c, Y_c / Z_c, np.ones_like(Z_c)))
        u = points_2d_hom[0, :]
        v = points_2d_hom[1, :]

        object_boxes_px = []
        for (x1, y1, x2, y2) in object_boxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Check if radar point (u[i], v[i]) lies inside box
            mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)

            depths_in_box = Z_c[mask]

            if len(depths_in_box) > 0:
                distance = np.median(depths_in_box)
            else:
                distance = np.nan

            distances.append(distance)
        return distances




