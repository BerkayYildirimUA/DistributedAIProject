import queue
import numpy as np
import cv2
import threading
import subprocess
import sys
import argparse
import carla
import math
import app.constants

from app.engine.world import World
from app.memory.shared_memory import RGBCameraMemory,DepthCameraMemory,VehicleDistanceMemory, VehicleStateMemory, RadarMemory, CameraCalibrationMemory


# Define transforms for handling camera data
def camera_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    new_frame = array[:, :, :3]
    frame_send_to_inference = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
    rgb_camera_memory.write(frame_send_to_inference)

# Callback to calculate depth map in meters
def depth_callback(image):
    test = "a"
    # array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    # b = array[:, :, 0].astype(np.float32)
    # g = array[:, :, 1].astype(np.float32)
    # r = array[:, :, 2].astype(np.float32)
    # normalized_depth = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0**3 - 1)
    # depth_meters = normalized_depth * 1000.0
    # depht_camera_memory.write(depth_meters)

# Radar callback (manual_control.py logic from PythonAPI/examples)
def radar_callback(radar_data):
    max_n = constants.RADAR_MAX_DETECTIONS

    # We will always write a (max_n, 5) array: [x, y, z, depth, velocity]
    # Zero rows mean "padding" and will be ignored, used for consistency in shared memory
    points = np.zeros((max_n, 5), dtype=np.float32)

    n = len(radar_data)
    if n == 0:
        radar_memory.write(points)  # all zeros
        return

    current_rot = radar_data.transform.rotation   # sensor rotation (world)
    sensor_loc  = radar_data.transform.location   # sensor location (world)

    # Fill up to max_n; clip if needed (shouldn't be necessary with 30k pps @ 20Hz)
    write_n = min(n, max_n)

    for i, det in enumerate(radar_data):
        if i >= write_n:
            break
        azi = math.degrees(det.azimuth)
        alt = math.degrees(det.altitude)

        # manual_control trick: pull the point a tad toward the sensor for visibility
        fw_vec = carla.Vector3D(x=det.depth - 0.25)

        # rotate local forward by (sensor rot + detection angles)
        carla.Transform(
            carla.Location(),
            carla.Rotation(
                pitch=current_rot.pitch + alt,
                yaw  =current_rot.yaw   + azi,
                roll =current_rot.roll)
        ).transform(fw_vec)

        # world point = sensor location + rotated forward vector
        world_point = sensor_loc + fw_vec

        points[i, 0] = world_point.x
        points[i, 1] = world_point.y
        points[i, 2] = world_point.z
        points[i, 3] = det.depth
        points[i, 4] = det.velocity

    radar_memory.write(points)

    P = world.calculate_camera_extrinsic()
    cam_mats[1, :, :] = P  # extrinsic, full 4x4 world -> camera (cv frame)
    camera_calibration_memory.write(cam_mats)


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
            test = "a"
            #depth_image = depth_camera_queue.get(timeout=1.0)
            #depth_callback(depth_image)
        except queue.Empty:
            continue
    
def process_radar_data():
    while True:
        try:
            raw_data = radar_queue.get(timeout=1.0)
            radar_callback(raw_data)
        except queue.Empty:
            continue



if __name__ == "__main__":
    # Create carla world and memory buffers
    world = World()
    rgb_camera_memory = RGBCameraMemory().get_write_access()
    #depht_camera_memory = DepthCameraMemory().get_write_access()
    vehicle_distance_memory = VehicleDistanceMemory().get_read_access()
    radar_memory = RadarMemory().get_write_access()
    camera_calibration_memory = CameraCalibrationMemory().get_write_access()
    rgb_camera_queue, radar_queue = world.expose_queues()

    K = world.calculate_camera_intrinsic()
    cam_mats = np.zeros((2, 4, 4), dtype=np.float64)
    cam_mats[0, :3, :3] = K  # intrinsic (3x3 in top-left corner)

    vehicle_state_memory = VehicleStateMemory().get_write_access()
    MAX_STEER_RAD = math.radians(60)  # ruwe schatting

    # Start threads
    rgb_thread = threading.Thread(target=process_rgb_images, daemon=True)
    depth_thread = threading.Thread(target=process_depth_images, daemon=True)
    radar_thread = threading.Thread(target=process_radar_data, daemon=True)
    rgb_thread.start()
    depth_thread.start()
    radar_thread.start()

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
    except KeyboardInterrupt:
        print("Closing simulation!")
    finally:
        world.cleanup()

        print("Cleanup complete.")




