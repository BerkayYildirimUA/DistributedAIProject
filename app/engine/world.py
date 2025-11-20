import queue
import random
import carla

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
        self.create_ego_cameras()
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
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta
        self.world.apply_settings(settings)

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

    def create_ego_cameras(self):
        camera_init_trans = carla.Transform(carla.Location(x=3, z=1.5), carla.Rotation(pitch=0, yaw=0, roll=0))
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "480")
        camera_bp.set_attribute("sensor_tick", "0.05")
        self.rgb_camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.ego_vehicle)
        self.rgb_camera_queue = queue.Queue(maxsize=10)
        self.rgb_camera.listen(lambda image: self.rgb_camera_queue.put_nowait(image))

        depth_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute("image_size_x", "640")
        depth_bp.set_attribute("image_size_y", "480")
        depth_bp.set_attribute("sensor_tick", "0.05")
        self.depth_camera = self.world.spawn_actor(depth_bp, camera_init_trans, attach_to=self.ego_vehicle)
        self.depth_camera_queue = queue.Queue(maxsize=10)
        self.depth_camera.listen(lambda image: self.depth_camera_queue.put_nowait(image))

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
        return self.rgb_camera_queue, self.depth_camera_queue

    def cleanup(self):
        # stop/destroy sensors
        try:
            self.rgb_camera.stop()
            self.rgb_camera.destroy()
        except Exception:
            pass
        try:
            self.depth_camera.stop()
            self.depth_camera.destroy()
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
