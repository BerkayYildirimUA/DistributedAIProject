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
RADAR_MAX_DETECTIONS = 50000

# Queue constants
QUEUE_MAXSIZE = 10

# Filename constants
ACTUAL_OBJECTS_IN_FRONT_COUNT_FILE = "actual_objects_in_front_count.gz"
ESTIMATED_OBJECT_IN_FRONT_COUNT_FILE = "estimated_objects_in_front_count.gz"
ACTUAL_VEHICLE_DISTANCE_IN_FRONT_FILE = "actual_objects_in_front_count.gz"
ESTIMATED_VEHICLE_DISTANCE_IN_FRONT_FILE = "estimated_vehicle_distance_in_front.gz"
