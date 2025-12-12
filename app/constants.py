import math

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

# Object detection constants
OBJECT_CLASS_NAMES = ["Vehicle", "Motor", "Bike","traffic light","traffic sign","pedestrian"]


# Statistics
# Filename constants
GT_OBJECTS_IN_FRONT_COUNT_FILE = "gt_objects_in_front_count.gz"
ESTIMATED_OBJECT_IN_FRONT_COUNT_FILE = "estimated_objects_in_front_count.gz"
GT_VEHICLE_DISTANCE_IN_FRONT_FILE = "gt_vehicle_distance_in_front.gz"
ESTIMATED_VEHICLE_DISTANCE_IN_FRONT_FILE = "estimated_vehicle_distance_in_front.gz"
GT_LEAD_DISTANCE_FILE="gt_lead_distance.gz"
LEAD_DISTANCE_FILE="lead_distance.gz"
GT_SPEED_LIMIT_FILE="gt_speed_limit.gz"
SPEED_FILE="speed.gz"
G_FORCE_FILE="g_force.gz"
GT_G_FORCE_FILE="gt_g_force.gz"

GT_SAFE_FOLLOWING_DISTANCE_FILE="gt_safe_following_distance.gz"
GT_TRAFFIC_SIGN_COUNT_FILE = "gt_traffic_signs_in_front_count.gz"
ESTIMATED_TRAFFIC_SIGN_COUNT_FILE = "estimated_traffic_signs_in_front_count.gz"
GT_TRAFFIC_LIGHT_COUNT_FILE = "gt_traffic_lights_in_front_count.gz"
ESTIMATED_TRAFFIC_LIGHT_COUNT_FILE = "estimated_traffic_lights_in_front_count.gz"
GT_VEHICLE_COUNT_FILE = "gt_vehicle_in_front_count.gz"
ESTIMATED_VEHICLE_COUNT_FILE = "estimated_vehicle_in_front_count.gz"
GT_PEDESTRIAN_COUNT_FILE = "gt_pedestrian_in_front_count.gz"
ESTIMATED_PEDESTRIAN_COUNT_FILE = "estimated_pedestrian_in_front_count.gz"

# Other
MAX_OBJECT_DETECT_DISTANCE = 20.0
MAX_LEAD_ACTOR_DISTANCE = 60.0
MAX_STEER_RAD=math.radians(60)
