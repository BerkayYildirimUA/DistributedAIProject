import numpy as np

# Sensor constants 
SENSOR_TICK = 0.05
SENSOR_POS_X = 1.5
SENSOR_POS_Z = 1.5
SENSOR_PITCH = 0
SENSOR_YAW = 0
SENSOR_ROLL = 0

# Camera sensor constants
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_ASPECT_RATIO = IMAGE_HEIGHT / IMAGE_WIDTH
HOR_FOV_DEG = 90
HOR_FOV_RAD = np.deg2rad(HOR_FOV_DEG)
VERT_FOV_RAD = 2 * np.arctan(IMAGE_ASPECT_RATIO * np.tan(HOR_FOV_RAD / 2))
VERT_FOV_DEG = np.rad2deg(VERT_FOV_RAD)

# Radar sensor constants
RADAR_RANGE = 250
RADAR_MAX_DETECTIONS = 100

# Queue constants
QUEUE_MAXSIZE = 10


# Precomputed Camera Intrinsics
focal_length = IMAGE_WIDTH / (2 * np.tan(HOR_FOV_RAD / 2))
CAMERA_INTRINSICS = np.array([
    [focal_length, 0, IMAGE_WIDTH / 2],
    [0, focal_length, IMAGE_HEIGHT / 2],
    [0, 0, 1]
])

# Precomputed Radar-to-Camera Transform
cy = np.cos(np.deg2rad(SENSOR_YAW))
sy = np.sin(np.deg2rad(SENSOR_YAW))
cp = np.cos(np.deg2rad(SENSOR_PITCH))
sp = np.sin(np.deg2rad(SENSOR_PITCH))
cr = np.cos(np.deg2rad(SENSOR_ROLL))
sr = np.sin(np.deg2rad(SENSOR_ROLL))

R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

ROTATION_MATRIX = R_yaw @ R_pitch @ R_roll

RADAR_TO_CAMERA_TRANSFORM = np.eye(4)
RADAR_TO_CAMERA_TRANSFORM[:3, :3] = ROTATION_MATRIX
RADAR_TO_CAMERA_TRANSFORM[:3, 3] = [SENSOR_POS_X, 0.0, SENSOR_POS_Z]