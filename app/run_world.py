import queue
import numpy as np
import cv2
import threading
import carla
import math
import constants


from engine.world import World
from memory.shared_memory import RGBCameraMemory,DepthCameraMemory,VehicleDistanceMemory,RadarMemory,CameraCalibrationMemory

# Create carla world and memory buffers
world = World()
rgb_camera_memory = RGBCameraMemory().get_write_access()
depht_camera_memory = DepthCameraMemory().get_write_access()
#vehicle_distance_memory = VehicleDistanceMemory().get_read_access()
radar_memory = RadarMemory().get_write_access()
camera_calibration_memory = CameraCalibrationMemory().get_write_access()
rgb_camera_queue, depth_camera_queue, radar_queue = world.expose_queues()
K, P = world.calculate_camera_intrinsic_extrinsic()
cam_mats = np.zeros((2, 4, 4), dtype=np.float64)
cam_mats[0, :3, :3] = K  # intrinsic (3x3 in top-left corner)
cam_mats[1, :, :] = P  # extrinsic, full 4x4 world -> camera (cv frame)
camera_calibration_memory.write(cam_mats)

# Define transforms for handling camera data
def camera_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    new_frame = array[:, :, :3]
    frame_send_to_inference = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
    rgb_camera_memory.write(frame_send_to_inference)

# Callback to calculate depth map in meters
def depth_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    b = array[:, :, 0].astype(np.float32)
    g = array[:, :, 1].astype(np.float32)
    r = array[:, :, 2].astype(np.float32)
    normalized_depth = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0**3 - 1)
    depth_meters = normalized_depth * 1000.0
    depht_camera_memory.write(depth_meters)

# Radar callback (manual_control.py logic from PythonAPI/examples)
def radar_callback(radar_data):

    n = len(radar_data)
    if n == 0:
        radar_memory.write(np.zeros((0, 4), dtype=np.float32))
        return

    # IMPORTANT: do NOT pre-allocate a fixed 100; use actual detections length
    points = np.zeros((n, 4), dtype=np.float32)

    current_rot = radar_data.transform.rotation  # sensor's world rotation this frame
    sensor_loc  = radar_data.transform.location  # sensor's world location this frame

    for i, detect in enumerate(radar_data):
        azi = math.degrees(detect.azimuth)
        alt = math.degrees(detect.altitude)

        # Manual-control trick: move points slightly toward the sensor to make dots visible
        fw_vec = carla.Vector3D(x=detect.depth - 0.25)

        # Rotate the forward vector by (sensor rotation + detection’s az/alt)
        carla.Transform(
            carla.Location(),
            carla.Rotation(
                pitch=current_rot.pitch + alt,
                yaw  =current_rot.yaw   + azi,
                roll =current_rot.roll)
        ).transform(fw_vec)

        # Convert to a world-space point by adding the sensor location
        world_point = sensor_loc + fw_vec

        # Store world XYZ + slant range (depth)
        points[i, 0] = world_point.x
        points[i, 1] = world_point.y
        points[i, 2] = world_point.z
        points[i, 3] = detect.depth

    radar_memory.write(points)




# ---------------------------
# Threaded data processing
# ---------------------------

def process_rgb_images():
    while True:
        try:
            image_carla = rgb_camera_queue.get(timeout=1.0)
            camera_callback(image_carla)
        except queue.Empty:
            continue

def process_depth_images():
    while True:
        try:
            depth_image = depth_camera_queue.get(timeout=1.0)
            depth_callback(depth_image)
        except queue.Empty:
            continue
    
def process_radar_data():
    while True:
        try:
            raw_data = radar_queue.get(timeout=1.0)
            radar_callback(raw_data)
        except queue.Empty:
            continue

# Start threads
rgb_thread = threading.Thread(target=process_rgb_images, daemon=True)
depth_thread = threading.Thread(target=process_depth_images, daemon=True)
radar_thread = threading.Thread(target=process_radar_data, daemon=True)
rgb_thread.start()
depth_thread.start()
radar_thread.start()


# Run the world
print("World started ticking!")
try:
    while True:
        try:
            world.tick()
        except RuntimeError as e:
            print(f"Tick failed {e}")

        # TODO: feed this distance data into the reinforcement module to calculate acceleration
        #distance_vehicle_in_front_m = vehicle_distance_memory[0,0]
        # print(f"Distance to vehicle in front: {distance_vehicle_in_front_m}m")
except KeyboardInterrupt:
    print("Closing simulation!")
finally:
    world.cleanup()






