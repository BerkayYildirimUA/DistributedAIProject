import queue
import random
import carla
import numpy as np
import cv2
import threading

# ---------------------------
# CARLA setup
# ---------------------------
client = carla.Client('localhost', 2000)
client.set_timeout(50.0)
world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = False
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)


client.load_world('Town05')
blueprint_library = world.get_blueprint_library()
vehicle_blueprints = blueprint_library.filter('*vehicle*')
ego_bp = blueprint_library.find('vehicle.tesla.model3')

# Get the map's spawn points
spawn_points = world.get_map().get_spawn_points()
# Spawn 50 vehicles randomly distributed throughout the map
# for each spawn point, we choose a random vehicle from the blueprint library
for i in range(0, 25):
    world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

spawned=False
while not spawned:
    try:
        ego_vehicle = world.spawn_actor(ego_bp, random.choice(spawn_points))
        spawned = True
    except:
        print("Trying other spawn location")


# ---------------
# RGB Camera
# ---------------

# Create a transform to place the camera on top of the vehicle
camera_init_trans = carla.Transform(carla.Location(z=1.5),carla.Rotation(pitch=0, yaw=0, roll=0))

# We create the camera through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute("image_size_x", "640")
camera_bp.set_attribute("image_size_y", "480")
camera_bp.set_attribute("sensor_tick", "0.05")
# camera_bp.set_attribute("sensor_tick", "0.05")  # match fixed_delta_seconds


# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# Create shared memory
filename = "shared_frame.dat"
image_shape = (480, 640, 3)
dtype = np.uint8
shared_frame = np.memmap(filename, dtype=dtype, mode='w+', shape=image_shape)
print('Shared memory for RGB camera created!')
def camera_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    new_frame = array[:, :, :3]
    frame_send_to_inference = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
    np.copyto(shared_frame, frame_send_to_inference)
    shared_frame.flush()

image_queue = queue.Queue(maxsize=10)
camera.listen(lambda image: image_queue.put_nowait(image))
# ---------------
# Depth Camera
# ---------------

# Depth camera setup
depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
depth_bp.set_attribute("image_size_x", "640")
depth_bp.set_attribute("image_size_y", "480")
depth_bp.set_attribute("sensor_tick", "0.05")
depth_cam = world.spawn_actor(depth_bp, camera_init_trans, attach_to=ego_vehicle)

# Create shared memory
depth_filename = "shared_depth.dat"
depth_shape = (480, 640)
depth_shared = np.memmap(depth_filename, dtype=np.float32, mode='w+', shape=depth_shape)
print('Shared memory for Depth Camera created!')

# Callback to calculate depth map in meters
def depth_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    b = array[:, :, 0].astype(np.float32)
    g = array[:, :, 1].astype(np.float32)
    r = array[:, :, 2].astype(np.float32)
    normalized_depth = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0**3 - 1)
    depth_meters = normalized_depth * 1000.0 
    np.copyto(depth_shared, depth_meters)
    depth_shared.flush()

depth_queue = queue.Queue(maxsize=10)
depth_cam.listen(lambda image: depth_queue.put_nowait(image))

# Shared memory for closest vehicle in front distance in meters
vehicle_distance_filename = "vehicle_distance.dat"
shared_vehicle_distance_in_front_m = np.memmap(vehicle_distance_filename, dtype=np.float32, mode='r', shape=(1,1))

# ---------------
# Ego vehicle
# ---------------

EgoLocation = ego_vehicle.get_location()

tm = client.get_trafficmanager()
tm.vehicle_percentage_speed_difference(ego_vehicle, 0)  # no speed scaling
tm.distance_to_leading_vehicle(ego_vehicle, 5.0)       # safety distance
tm.ignore_vehicles_percentage(ego_vehicle, 0)          # don’t ignore virtual vehicles
tm.max_speed(ego_vehicle, 20.0) 
for vehicle in world.get_actors().filter('*vehicle*'):
    if vehicle.id != ego_vehicle.id:
        vehicle.set_autopilot(True,tm.get_port())
ego_vehicle.set_autopilot(True, tm.get_port())



# ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))

# ---------------------------
# Threaded image processing
# ---------------------------

def process_rgb_images():
    while True:
        try:
            image_carla = image_queue.get(timeout=1.0)
            camera_callback(image_carla)
        except queue.Empty:
            continue

def process_depth_images():
    while True:
        try:
            depth_image = depth_queue.get(timeout=1.0)
            depth_callback(depth_image)
        except queue.Empty:
            continue

# Start threads
rgb_thread = threading.Thread(target=process_rgb_images, daemon=True)
depth_thread = threading.Thread(target=process_depth_images, daemon=True)
rgb_thread.start()
depth_thread.start()


spectator = world.get_spectator()
print("World started ticking!")
try:
    while True:
        try:
            world.tick()
        except RuntimeError as e:
            print(f"Tick failed {e}")
        transform = ego_vehicle.get_transform()
        # Compute position 10m behind and 5m above ego car
        forward_vector = transform.get_forward_vector()
        spectator_location = transform.location - 10 * forward_vector + carla.Location(z=5)
        spectator_transform = carla.Transform(spectator_location, transform.rotation)
        spectator.set_transform(spectator_transform)

        # TODO: feed this distance data into the reinforcement module to calculate acceleration
        distance_vehicle_in_front_m = shared_vehicle_distance_in_front_m[0,0]
        print(f"Distance to vehicle in front: {distance_vehicle_in_front_m}m")
except KeyboardInterrupt:
    print("Closing simulation!")
finally:
    camera.stop()
    camera.destroy()
    depth_cam.stop()
    depth_cam.destroy()
    ego_vehicle.destroy()






