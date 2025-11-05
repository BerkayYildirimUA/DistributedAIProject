import queue
import random
import carla
import constants

class World:
    def __init__(self):
        # Parameters
        self.port=2000
        self.timeout=50.0
        self.world_name="Town05"
        self.delta=0.05

        self.init()

    def init(self):
        # Create world
        self.create_world()
        # Spawn random vehicles
        self.spawn_random_vehicles()
        # Spawn ego vehicle
        self.create_and_spawn_ego_vehicle()
        # Enable autopilot
        self.enable_autopilot_for_ego_vehicle()
        # Create cameras and attach to ego vehicle
        self.create_ego_sensors()
        # Set spectator
        self.spectator = self.world.get_spectator()

    def tick(self):
        self.world.tick()
        # Update spectator view
        self.update_spectator()

    def create_world(self):
        self.client = carla.Client('localhost', self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = self.delta
        self.world.apply_settings(settings)
        self.client.load_world(self.world_name)


    def get_vehicle_bps(self):
        blueprint_library = self.world.get_blueprint_library()
        return blueprint_library.filter('*vehicle*')

    def get_ego_vehicle_bps(self):
        return self.get_vehicle_bps().find('vehicle.tesla.model3')

    def spawn_random_vehicles(self):
        # Get the map's spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        # Spawn 50 vehicles randomly distributed throughout the map
        # for each spawn point, we choose a random vehicle from the blueprint library
        for i in range(0, 50):
            self.world.try_spawn_actor(random.choice(self.get_vehicle_bps()), random.choice(spawn_points))

    def create_and_spawn_ego_vehicle(self):
        spawn_points = self.world.get_map().get_spawn_points()
        spawned = False
        max_tries=100
        while not spawned:
            try:
                self.ego_vehicle = self.world.spawn_actor(self.get_ego_vehicle_bps(), random.choice(spawn_points))
                spawned = True
            except:
                print("Trying other spawn location")
                max_tries-=1
                if max_tries<=0:
                    raise Exception("Failed to spawn ego vehicle")

    def create_ego_sensors(self):
        sensor_location = carla.Location(x=constants.SENSOR_POS_X, z=constants.SENSOR_POS_Z)
        sensor_rotation = carla.Rotation(pitch=constants.SENSOR_PITCH, yaw=constants.SENSOR_YAW, roll=constants.SENSOR_ROLL)
        camera_init_trans = carla.Transform(sensor_location, sensor_rotation)
        
        # We create the camera through a blueprint that defines its properties
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", str(constants.IMAGE_WIDTH))
        camera_bp.set_attribute("image_size_y", str(constants.IMAGE_HEIGHT))
        camera_bp.set_attribute("sensor_tick", str(constants.SENSOR_TICK))
        camera_bp.set_attribute("fov", str(constants.HOR_FOV_DEG))
        # We spawn the camera and attach it to our ego vehicle
        self.rgb_camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.ego_vehicle)
        self.rgb_camera_queue = queue.Queue(maxsize=constants.QUEUE_MAXSIZE)
        self.rgb_camera.listen(lambda image: self.rgb_camera_queue.put_nowait(image))

        # Depth camera setup
        # TODO: change max depth value to a value found in real depth camera setups
        depth_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute("image_size_x", str(constants.IMAGE_WIDTH))
        depth_bp.set_attribute("image_size_y", str(constants.IMAGE_HEIGHT))
        depth_bp.set_attribute("sensor_tick", str(constants.SENSOR_TICK))
        depth_bp.set_attribute("fov", str(constants.HOR_FOV_DEG))
        self.depth_camera = self.world.spawn_actor(depth_bp, camera_init_trans, attach_to=self.ego_vehicle)
        self.depth_camera_queue = queue.Queue(maxsize=constants.QUEUE_MAXSIZE)
        self.depth_camera.listen(lambda image: self.depth_camera_queue.put_nowait(image))

        # Radar setup
        blueprint_library = self.world.get_blueprint_library()
        radar_bp = blueprint_library.find('sensor.other.radar')
        # TODO: change these parameters to values found in real radar setups
        radar_bp.set_attribute('horizontal_fov', str(constants.HOR_FOV_DEG))  
        radar_bp.set_attribute('vertical_fov', str(constants.VERT_FOV_DEG))    
        radar_bp.set_attribute('range', str(constants.RADAR_RANGE))         
        radar_transform = carla.Transform(sensor_location, sensor_rotation)
        self.radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.ego_vehicle)
        self.radar_queue = queue.Queue(maxsize=constants.QUEUE_MAXSIZE)
        # check if queue is full: yes --> pop oldest, push new one. no --> push. Ensures most recent radar data is in the queue
        self.radar.listen(lambda data: (self.radar_queue.get_nowait(), self.radar_queue.put_nowait(data)) if self.radar_queue.full() else self.radar_queue.put_nowait(data))

        print("Camera Transform:", self.rgb_camera.get_transform())
        print("Radar Transform:", self.radar.get_transform())

    def enable_autopilot_for_ego_vehicle(self):
        traffic_manager = self.client.get_trafficmanager()
        for vehicle in self.world.get_actors().filter('*vehicle*'):
            if vehicle.id != self.ego_vehicle.id:
                vehicle.set_autopilot(True, traffic_manager.get_port())
        self.ego_vehicle.set_autopilot(True, traffic_manager.get_port())

    def update_spectator(self):
        transform = self.ego_vehicle.get_transform()
        # Compute position 10m behind and 5m above ego car
        forward_vector = transform.get_forward_vector()
        spectator_location = transform.location - 10 * forward_vector + carla.Location(z=5)
        spectator_transform = carla.Transform(spectator_location, transform.rotation)
        self.spectator.set_transform(spectator_transform)

    def expose_queues(self):
        return self.rgb_camera_queue, self.depth_camera_queue, self.radar_queue

    def cleanup(self):
        self.rgb_camera.stop()
        self.rgb_camera.destroy()
        self.depth_camera.stop()
        self.depth_camera.destroy()
        self.ego_vehicle.destroy()