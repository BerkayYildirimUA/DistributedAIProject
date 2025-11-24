import queue
import random
import carla
import constants
import math
import numpy as np

class World:
    def __init__(self):
        # Parameters
        self.port = 2000
        self.timeout = 50.0
        self.world_name = "Town05"
        self.delta = 0.05

        # holders for cleanup
        self.walkers = []
        self.walker_controllers = []

        self.init()

    def init(self):
        # Create world
        self.create_world()
        # Spawn random vehicles
        self.spawn_random_vehicles()
        # NEW: spawn pedestrians (walkers)
        #self.spawn_pedestrians(num_walkers=40)
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

        self.client.load_world(self.world_name)
        self.world = self.client.get_world()

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        # settings.fixed_delta_seconds = self.delta
        self.world.apply_settings(settings)
        self.client.load_world(self.world_name)

    def get_vehicle_bps(self):
        blueprint_library = self.world.get_blueprint_library()
        return blueprint_library.filter('*vehicle*')

    def get_ego_vehicle_bps(self):
        return self.world.get_blueprint_library().find('vehicle.tesla.model3')

    def spawn_random_vehicles(self):
        spawn_points = self.world.get_map().get_spawn_points()
        for _ in range(25):
            self.world.try_spawn_actor(random.choice(self.get_vehicle_bps()), random.choice(spawn_points))

    # -------------------------
    # NEW: pedestrians (walkers)
    # -------------------------
    def spawn_pedestrians(self, num_walkers=40):
        bp_lib = self.world.get_blueprint_library()
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        controller_bp = bp_lib.find('controller.ai.walker')

        spawned = 0
        while spawned < num_walkers:
            # kies een random nav-mesh locatie
            loc = self.world.get_random_location_from_navigation()
            if not loc:
                continue
            w_bp = random.choice(walker_bps)

            # maak ze niet onsterfelijk zodat ze reageren op verkeer
            if w_bp.has_attribute('is_invincible'):
                w_bp.set_attribute('is_invincible', 'false')

            try:
                walker = self.world.spawn_actor(w_bp, carla.Transform(loc))
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)

                # start eenvoudige AI: loop naar random bestemming met normale loopsnelheid
                controller.start()
                controller.go_to_location(self.world.get_random_location_from_navigation())
                controller.set_max_speed(1.4)  # ~1.4 m/s = normale wandeltempo

                self.walkers.append(walker)
                self.walker_controllers.append(controller)
                spawned += 1
            except RuntimeError:
                # probeer gewoon een andere locatie
                continue

    def create_and_spawn_ego_vehicle(self):
        spawn_points = self.world.get_map().get_spawn_points()
        spawned = False
        max_tries = 100
        while not spawned and max_tries > 0:
            try:
                self.ego_vehicle = self.world.spawn_actor(self.get_ego_vehicle_bps(), random.choice(spawn_points))
                spawned = True
            except Exception:
                max_tries -= 1
                if max_tries <= 0:
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
        #self.rgb_camera.listen(lambda image: self.rgb_camera_queue.put_nowait(image))
        self.rgb_camera.listen(lambda data: (self.rgb_camera_queue.get_nowait(), self.rgb_camera_queue.put_nowait(data)) if self.rgb_camera_queue.full() else self.rgb_camera_queue.put_nowait(data))


        # Depth camera setup
        # TODO: change max depth value to a value found in real depth camera setups
        # depth_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        # depth_bp.set_attribute("image_size_x", str(constants.IMAGE_WIDTH))
        # depth_bp.set_attribute("image_size_y", str(constants.IMAGE_HEIGHT))
        # depth_bp.set_attribute("sensor_tick", str(constants.SENSOR_TICK))
        # depth_bp.set_attribute("fov", str(constants.HOR_FOV_DEG))
        # self.depth_camera = self.world.spawn_actor(depth_bp, camera_init_trans, attach_to=self.ego_vehicle)
        # self.depth_camera_queue = queue.Queue(maxsize=constants.QUEUE_MAXSIZE)
        # self.depth_camera.listen(lambda image: self.depth_camera_queue.put_nowait(image))

        # Radar setup
        blueprint_library = self.world.get_blueprint_library()
        radar_bp = blueprint_library.find('sensor.other.radar')
        # TODO: change these parameters to values found in real radar setups
        radar_bp.set_attribute('horizontal_fov', str(constants.HOR_FOV_DEG))  
        radar_bp.set_attribute('vertical_fov', str(constants.VERT_FOV_DEG))    
        radar_bp.set_attribute('range', str(constants.RADAR_RANGE))
        radar_bp.set_attribute('points_per_second', '30000')
        radar_bp.set_attribute('sensor_tick', str(constants.SENSOR_TICK))
        radar_transform = carla.Transform(sensor_location, sensor_rotation)
        self.radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.ego_vehicle)
        self.radar_queue = queue.Queue(maxsize=constants.QUEUE_MAXSIZE)
        # check if queue is full: yes --> pop oldest, push new one. no --> push. Ensures most recent radar data is in the queue
        self.radar.listen(lambda data: (self.radar_queue.get_nowait(), self.radar_queue.put_nowait(data)) if self.radar_queue.full() else self.radar_queue.put_nowait(data))

        print("Camera attrs:", self.rgb_camera.attributes)
        print("Radar attrs:", self.radar.attributes)

    def enable_autopilot_for_ego_vehicle(self):
        traffic_manager = self.client.get_trafficmanager()
        for vehicle in self.world.get_actors().filter('*vehicle*'):
            if vehicle.id != self.ego_vehicle.id:
                vehicle.set_autopilot(True, traffic_manager.get_port())
        self.ego_vehicle.set_autopilot(True, traffic_manager.get_port())

    def update_spectator(self):
        transform = self.ego_vehicle.get_transform()
        forward_vector = transform.get_forward_vector()
        spectator_location = transform.location - 10 * forward_vector + carla.Location(z=5)
        spectator_transform = carla.Transform(spectator_location, transform.rotation)
        self.spectator.set_transform(spectator_transform)

    def expose_queues(self):
        return self.rgb_camera_queue, self.radar_queue

    def calculate_camera_extrinsic(self):
        # World -> camera in Unreal frame (X forward, Y right, Z up)
        T_world_cam_ue = np.array(self.rgb_camera.get_transform().get_inverse_matrix(),
                                  dtype=np.float64)  # (4,4)

        # Unreal -> CV frame (x right, y down, z forward)
        R_ue2cv = np.array([[0, 1, 0],
                            [0, 0, -1],
                            [1, 0, 0]], dtype=np.float64)
        T_ue2cv = np.eye(4, dtype=np.float64)
        T_ue2cv[:3, :3] = R_ue2cv

        # Final world -> camera (CV frame)
        P = T_ue2cv @ T_world_cam_ue  # (4,4)
        return P

    def calculate_camera_intrinsic(self):
        w = float(constants.IMAGE_WIDTH)
        h = float(constants.IMAGE_HEIGHT)
        hfov = math.radians(constants.HOR_FOV_DEG)

        # Intrinsics
        fx = w / (2.0 * math.tan(hfov / 2.0))
        # exact fy based on aspect
        vfov = 2.0 * math.atan((h / w) * math.tan(hfov / 2.0))
        fy = h / (2.0 * math.tan(vfov / 2.0))
        cx = (w - 1.0) / 2.0
        cy = (h - 1.0) / 2.0

        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        return K

    def cleanup(self):
        # stop/destroy sensors
        try:
            self.rgb_camera.stop()
            self.rgb_camera.destroy()
        except Exception:
            pass
        #try:
        #    self.depth_camera.stop()
        #    self.depth_camera.destroy()
        except Exception:
            pass

        # stop/destroy pedestrian controllers first
        for c in self.walker_controllers:
            try:
                c.stop()
            except Exception:
                pass
        for c in self.walker_controllers:
            try:
                c.destroy()
            except Exception:
                pass

        # then destroy walkers
        for w in self.walkers:
            try:
                w.destroy()
            except Exception:
                pass

        # # finally ego vehicle
        # try:
        #     self.ego_vehicle.destroy()
        # except Exception:
        #     pass
