import queue
import carla
import constants
import time
import math

class World:
    def __init__(self):
        self.port = 2000
        self.timeout = 50.0
        self.world_name = "Town05"
        self.delta = 0.05
        self.forward_speed = 0.1     # m/s
        self.reverse_speed = -0.1    # m/s
        self.move_duration = 2.0     # seconds forward/backward per cycle
        self.last_direction_change = time.time()
        self.moving_forward = True

        self.init()

    def init(self):
        self.create_world()
        self.spawn_ego_vehicle()
        self.spawn_static_vehicles()
        self.create_ego_sensors()
        self.spectator = self.world.get_spectator()

    def create_world(self):
        self.client = carla.Client('localhost', self.port)
        self.client.set_timeout(self.timeout)
        self.client.load_world(self.world_name)
        self.world = self.client.get_world()

        settings = self.world.get_settings()
        settings.synchronous_mode = False  # for simplicity
        settings.fixed_delta_seconds = self.delta
        self.world.apply_settings(settings)

    def spawn_ego_vehicle(self):
        bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]
        self.ego_vehicle = self.world.spawn_actor(bp, spawn_point)
        print(f"Ego vehicle spawned at {spawn_point.location}")

    def spawn_static_vehicles(self):
        bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        ego_tf = self.ego_vehicle.get_transform()
        fwd = ego_tf.get_forward_vector()

        # Spawn two static vehicles in front of ego
        offsets = [10.0, 25.0]  # meters ahead
        self.static_vehicles = []
        for i, offset in enumerate(offsets):
            spawn_loc = ego_tf.location + fwd * offset
            spawn_tf = carla.Transform(spawn_loc, ego_tf.rotation)
            vehicle = self.world.spawn_actor(bp, spawn_tf)
            vehicle.set_autopilot(False)
            vehicle.set_simulate_physics(True)
            self.static_vehicles.append(vehicle)
            print(f"Spawned static vehicle {i + 1} at distance {offset} m")

    def create_ego_sensors(self):
        sensor_location = carla.Location(x=constants.SENSOR_POS_X, z=constants.SENSOR_POS_Z)
        sensor_rotation = carla.Rotation(pitch=constants.SENSOR_PITCH, yaw=constants.SENSOR_YAW, roll=constants.SENSOR_ROLL)
        camera_init_trans = carla.Transform(sensor_location, sensor_rotation)

        # RGB Camera
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", str(constants.IMAGE_WIDTH))
        camera_bp.set_attribute("image_size_y", str(constants.IMAGE_HEIGHT))
        camera_bp.set_attribute("sensor_tick", str(constants.SENSOR_TICK))
        camera_bp.set_attribute("fov", str(constants.HOR_FOV_DEG))
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
        self.depth_camera.listen(lambda image: (
            self.depth_camera_queue.get_nowait(), self.depth_camera_queue.put_nowait(image))
            if self.depth_camera_queue.full() else self.depth_camera_queue.put_nowait(image)
        )

        # Radar
        radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', str(constants.HOR_FOV_DEG))
        radar_bp.set_attribute('vertical_fov', str(constants.VERT_FOV_DEG))
        radar_bp.set_attribute('range', str(constants.RADAR_RANGE))
        radar_transform = carla.Transform(sensor_location, sensor_rotation)
        self.radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.ego_vehicle)
        self.radar_queue = queue.Queue(maxsize=constants.QUEUE_MAXSIZE)
        self.radar.listen(lambda data: (
            self.radar_queue.get_nowait(), self.radar_queue.put_nowait(data))
            if self.radar_queue.full() else self.radar_queue.put_nowait(data)
        )

    def tick(self):
        self.world.tick()
        self.update_motion()
        self.update_spectator()

    def update_motion(self):
        now = time.time()
        if now - self.last_direction_change > self.move_duration:
            self.moving_forward = not self.moving_forward
            self.last_direction_change = now

        control = carla.VehicleControl()
        control.throttle = 0.4 if self.moving_forward else 0.4
        control.reverse = not self.moving_forward
        control.steer = 0.0
        self.ego_vehicle.apply_control(control)

    def update_spectator(self):
        transform = self.ego_vehicle.get_transform()
        forward_vector = transform.get_forward_vector()
        spectator_location = transform.location - 10 * forward_vector + carla.Location(z=5)
        spectator_transform = carla.Transform(spectator_location, transform.rotation)
        self.spectator.set_transform(spectator_transform)

    def expose_queues(self):
        return self.rgb_camera_queue, self.radar_queue, self.depth_camera_queue

    def cleanup(self):
        actors = [self.rgb_camera, self.radar, self.ego_vehicle] + self.static_vehicles
        for actor in actors:
            if actor is not None:
                actor.destroy()
        print("Cleaned up all actors.")
