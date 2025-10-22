import queue
import numpy as np
import cv2
import threading

from engine.world import World
from memory.shared_memory import RGBCameraMemory,DepthCameraMemory,VehicleDistanceMemory

# Create carla world and memory buffers
world = World()
rgb_camera_memory = RGBCameraMemory().get_write_access()
depht_camera_memory = DepthCameraMemory().get_write_access()
vehicle_distance_memory = VehicleDistanceMemory().get_read_access()
rgb_camera_queue, depth_camera_queue = world.expose_queues()


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

# ---------------------------
# Threaded image processing
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

# Start threads
rgb_thread = threading.Thread(target=process_rgb_images, daemon=True)
depth_thread = threading.Thread(target=process_depth_images, daemon=True)
rgb_thread.start()
depth_thread.start()


# Run the world
print("World started ticking!")
try:
    while True:
        try:
            world.tick()
        except RuntimeError as e:
            print(f"Tick failed {e}")

        # TODO: feed this distance data into the reinforcement module to calculate acceleration
        distance_vehicle_in_front_m = vehicle_distance_memory[0,0]
        print(f"Distance to vehicle in front: {distance_vehicle_in_front_m}m")
except KeyboardInterrupt:
    print("Closing simulation!")
finally:
    world.cleanup()






