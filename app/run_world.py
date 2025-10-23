import queue
import numpy as np
import cv2
import threading
import subprocess
import sys
import argparse

from engine.world import World
from memory.shared_memory import RGBCameraMemory,DepthCameraMemory,VehicleDistanceMemory



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

    #parse arguments
    parser = argparse.ArgumentParser(description="Run CARLA simulation bridge and optionally launch another script.")
    parser.add_argument(
        '--interpreter',
        metavar='PATH',
        type=str,
        help='Path to the Python interpreter for the second script (e.g., run_vehicle_pov.py). Defaults to the current interpreter.'
    )
    args = parser.parse_args()

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

    if args.interpreter:
        other_python_interpreter = args.interpreter # Use the modern interpreter
        print(f"Using provided interpreter: {other_python_interpreter}")
    else:
        other_python_interpreter = sys.executable  # Use the current interpreter
        print(f"Using current interpreter: {other_python_interpreter}")

    script_to_run = "run_vehicle_pov.py"

    tick_counter = 0
    process_launched = False
    pov_process = None

    try:
        while True:
            try:
                world.tick()
                if tick_counter < 25:
                    tick_counter += 1
            except RuntimeError as e:
                print(f"Tick failed {e}")

            # --- Subprocess Launch ---
            if not process_launched and tick_counter >= 20:
                print(f"\n[Tick {tick_counter}] Reached 20 ticks. Launching '{script_to_run}'...")
                try:
                    pov_process = subprocess.Popen(
                        [other_python_interpreter, script_to_run],
                        stdout=sys.stdout,
                        stderr=sys.stderr
                    )
                    print(f"Launched process with PID: {pov_process.pid}")
                    process_launched = True  # Set flag so it doesn't run again
                except FileNotFoundError:
                    print(f"ERROR: Could not find the interpreter '{other_python_interpreter}'")
                    print("Please check the path and try again.")
                    break  # Exit main loop
                except Exception as e:
                    print(f"Failed to launch process: {e}")
                    break  # Exit main loop
                print("--------------------------------------------------\n")
            # --- End Launch ---

            # TODO: feed this distance data into the reinforcement module to calculate acceleration
            distance_vehicle_in_front_m = vehicle_distance_memory[0, 0]
            # print(f"Distance to vehicle in front: {distance_vehicle_in_front_m}m")
    except KeyboardInterrupt:
        print("Closing simulation!")
    finally:
        world.cleanup()
        if pov_process and pov_process.poll() is None:  # Check if process exists and is running
            print(f"Terminating subprocess PID: {pov_process.pid}...")
            pov_process.terminate()
            try:
                pov_process.wait(timeout=5)  # Wait a bit for graceful termination
                print("Subprocess terminated.")
            except subprocess.TimeoutExpired:
                print("Subprocess did not terminate gracefully, killing.")
                pov_process.kill()
                print("Subprocess killed.")
        print("Cleanup complete.")




