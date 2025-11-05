import numpy as np

# Sensor constants 
SENSOR_TICK = 0.05
SENSOR_POS_X = 1.5
SENSOR_POS_Z = 1.5
SENSOR_PITCH = 0
SENSOR_YAW = 0
SENSOR_ROLL = 0
SENSOR_PITCH_RAD = np.deg2rad(SENSOR_PITCH)
SENSOR_YAW_RAD = np.deg2rad(SENSOR_YAW)
SENSOR_ROLL_RAD = np.deg2rad(SENSOR_ROLL)

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

# Camera - Radar constants, used for distance calculation
F_X = IMAGE_WIDTH / (2 * np.tan(HOR_FOV_RAD / 2))
F_Y = F_X  # assuming square pixels, same focal length
C_X = IMAGE_WIDTH / 2
C_Y = IMAGE_HEIGHT / 2

K = np.array([
    [F_X, 0.0, C_X],
    [0.0, F_Y, C_Y],
    [0.0, 0.0, 1.0]
])

# Separate rotation matrices
RX = np.array([[1, 0, 0],
               [0, np.cos(SENSOR_ROLL_RAD), -np.sin(SENSOR_ROLL_RAD)],
               [0, np.sin(SENSOR_ROLL_RAD),  np.cos(SENSOR_ROLL_RAD)]])
RY = np.array([[ np.cos(SENSOR_PITCH_RAD), 0, np.sin(SENSOR_PITCH_RAD)],
               [0, 1, 0],
               [-np.sin(SENSOR_PITCH_RAD), 0, np.cos(SENSOR_PITCH_RAD)]])
RZ = np.array([[ np.cos(SENSOR_YAW_RAD), -np.sin(SENSOR_YAW_RAD), 0],
               [ np.sin(SENSOR_YAW_RAD),  np.cos(SENSOR_YAW_RAD), 0],
               [0, 0, 1]])

# Combined rotation: roll → pitch → yaw
CAM_ROT = RZ @ RY @ RX

CAM_T = np.array([0.0, 0.0, 0.0])

CAM_EXTRINSIC = np.eye(4)
CAM_EXTRINSIC[:3,:3] = CAM_ROT
CAM_EXTRINSIC[:3,3] = CAM_T
