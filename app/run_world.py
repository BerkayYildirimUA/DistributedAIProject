import queue
import numpy as np
import cv2
import threading
import subprocess
import sys
import argparse

from engine.world import World
from memory.shared_memory import RGBCameraMemory, DepthCameraMemory, VehicleDistanceMemory, VehicleStateMemory
import math


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



if __name__ == "__main__":

    vehicle_state_memory = VehicleStateMemory().get_write_access()
    MAX_STEER_RAD = math.radians(60)  # ruwe schatting


    # Create carla world and memory buffers
    world = World()
    rgb_camera_memory = RGBCameraMemory().get_write_access()
    depht_camera_memory = DepthCameraMemory().get_write_access()
    vehicle_distance_memory = VehicleDistanceMemory().get_read_access()
    rgb_camera_queue, depth_camera_queue = world.expose_queues()

    # Start threads
    rgb_thread = threading.Thread(target=process_rgb_images, daemon=True)
    depth_thread = threading.Thread(target=process_depth_images, daemon=True)
    rgb_thread.start()
    depth_thread.start()

    try:
        while True:
            try:
                world.tick()
                # --- we get the state of the vehicle and put into shared memory ---
                vel = world.ego_vehicle.get_velocity()                          # get the velocity from our car in CARLA
                speed_ms = float((vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5) # calculate the speed

                ctrl = world.ego_vehicle.get_control()                          # get the control applied in the last tick
                # ctrl.steer in [-1,1] => schaal naar rad
                steer_rad = -float(ctrl.steer) * MAX_STEER_RAD                  # calculating the steer angle

                vehicle_state_memory.write(np.array([speed_ms, steer_rad], dtype=np.float32))
                # --------------------------------------

            except RuntimeError as e:
                print(f"Tick failed {e}")


            # TODO: feed this distance data into the reinforcement module to calculate acceleration
            distance_vehicle_in_front_m = vehicle_distance_memory[0, 0]
            # print(f"Distance to vehicle in front: {distance_vehicle_in_front_m}m")
    except KeyboardInterrupt:
        print("Closing simulation!")
    finally:
        world.cleanup()

        print("Cleanup complete.")




