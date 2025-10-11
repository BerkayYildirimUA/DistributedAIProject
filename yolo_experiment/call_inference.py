import queue
import random
import carla
import numpy as np
import cv2

# ---------------------------
# CARLA setup
# ---------------------------
client = carla.Client('localhost', 2000)
client.set_timeout(50.0)
world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True
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
for i in range(0,50):
    world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

spawned=False
while not spawned:
    try:
        ego_vehicle = world.spawn_actor(ego_bp, random.choice(spawn_points))
        spawned = True
    except:
        print("Trying other spawn location")

# Create a transform to place the camera on top of the vehicle
camera_init_trans = carla.Transform(carla.Location(z=1.5),carla.Rotation(pitch=0, yaw=0, roll=0))

# We create the camera through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute("image_size_x", "640")
camera_bp.set_attribute("image_size_y", "480")
camera_bp.set_attribute("sensor_tick", "0.033")
# camera_bp.set_attribute("sensor_tick", "0.05")  # match fixed_delta_seconds


# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# Create shared memory
filename = "shared_frame.dat"
image_shape = (480, 640, 3)
dtype = np.uint8
shared_frame = np.memmap(filename, dtype=dtype, mode='w+', shape=image_shape)
print('Memory created!')
def camera_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    new_frame = array[:, :, :3]
    frame_send_to_inference = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
    np.copyto(shared_frame, frame_send_to_inference)
    shared_frame.flush()

image_queue = queue.Queue()
camera.listen(image_queue.put)
EgoLocation = ego_vehicle.get_location()

traffic_manager = client.get_trafficmanager()
for vehicle in world.get_actors().filter('*vehicle*'):
    if vehicle.id != ego_vehicle.id:
        vehicle.set_autopilot(True,traffic_manager.get_port())
ego_vehicle.set_autopilot(True, traffic_manager.get_port())

# ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))

spectator = world.get_spectator()
print("World started ticking!")
try:
    while True:
        world.tick()
        transform = ego_vehicle.get_transform()
        # Compute position 10m behind and 5m above ego car
        forward_vector = transform.get_forward_vector()
        spectator_location = transform.location - 10 * forward_vector + carla.Location(z=5)

        spectator_transform = carla.Transform(spectator_location, transform.rotation)
        spectator.set_transform(spectator_transform)
        try:
            image_carla = image_queue.get_nowait()
            camera_callback(image_carla)
        except queue.Empty:
            continue
except KeyboardInterrupt:
    print("Closing simulation!")
finally:
    camera.stop()
    camera.destroy()
    ego_vehicle.destroy()





