import math
import queue
import random
import weakref
import carla
import numpy as np

from memory.shared_memory import RadarMemory

import numpy as np
import carla


def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def transform_to_matrix(transform: carla.Transform):
    """
    Convert carla.Transform to 4x4 homogeneous transformation matrix.
    """
    # Convert rotation from degrees to radians
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw   = np.deg2rad(transform.rotation.yaw)
    roll  = np.deg2rad(transform.rotation.roll)

    # Rotation matrices around x, y, z axes
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [ 0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # CARLA uses yaw-pitch-roll (Z-Y-X) order
    R = Rz @ Ry @ Rx

    # Translation vector
    T = np.array([transform.location.x,
                  transform.location.y,
                  transform.location.z]).reshape(3,1)

    # Homogeneous 4x4 matrix
    H = np.eye(4)
    H[:3,:3] = R
    H[:3,3] = T.flatten()
    return H



class RadarSensor(object):
    def __init__(self, parent_actor, camera_actor,camera_transform):
        self.sensor = None
        self._parent = parent_actor
        self._camera = camera_actor
        self.velocity_range = 7.5  # m/s
        world = self._parent.get_world()

        self.radar_memory = RadarMemory().get_write_access()

        # Get radar blueprint
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', '35')  # match camera FOV
        bp.set_attribute('vertical_fov', '20')
        bp.set_attribute('range', '50')  # max detection range in meters

        # Recommended radar transform
        radar_transform = carla.Transform(
            carla.Location(x=2.0, y=0.0, z=1.5),  # 2m forward, 1.5m high, centered
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)  # aligned with camera
        )

        # Spawn radar attached to the ego vehicle
        self.sensor = world.spawn_actor(
            bp,
            radar_transform,
            attach_to=self._parent
        )


        # Build camera intrinsics
        image_w = int(self._camera.attributes['image_size_x'])
        image_h = int(self._camera.attributes['image_size_y'])
        fov = int(self._camera.attributes['fov'])

        self.camera_transform = camera_transform
        self.img_w, self.img_h = image_w, image_h

        # compute focal length in pixels
        f_x = f_y = image_w / (2 * np.tan(np.deg2rad(fov) / 2))
        c_x = image_w / 2
        c_y = image_h / 2

        self.K = np.array([[f_x, 0, c_x],
                      [0, f_y, c_y],
                      [0, 0, 1]])

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda data: RadarSensor._Radar_callback(weak_self, data))

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()


    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        data=[]
        self = weak_self()
        if not self:
            return

        for detect in radar_data:
            # Convert polar to Cartesian coordinates (flatten to ground plane)
            x = detect.depth * math.cos(detect.altitude) * math.cos(detect.azimuth)
            y = detect.depth * math.cos(detect.altitude) * math.sin(detect.azimuth)
            z = detect.depth * math.sin(detect.altitude)
            # z = 0.0  # flatten to ground plane
            point_radar = np.array([x, y, z, 1.0]).reshape(4, 1)

            H_radar = transform_to_matrix(self.radar_transform)
            H_camera = transform_to_matrix(self.camera_transform)
            H_radar_to_camera = np.linalg.inv(H_camera) @ H_radar
            R_radar_to_camera = H_radar_to_camera[:3, :3]
            t_radar_to_camera = H_radar_to_camera[:3, 3]
            point_camera = H_radar_to_camera @ point_radar
            X_cam, Y_cam, Z_cam = point_camera[:3, 0]

            u = (self.K[0, 0] * X_cam + self.K[0, 2] * Z_cam) / Z_cam
            v = (self.K[1, 1] * Y_cam + self.K[1, 2] * Z_cam) / Z_cam

            data.append([u, v, detect.velocity])

        max_points = 500
        num_points = len(data)

        if num_points < max_points:
            # append [0,0,0] lists
            data += [[0, 0, 0]] * (max_points - num_points)
        else:
            data = data[:max_points]
        self.radar_memory.write(data)


# class RadarSensor(object):
#     def __init__(self, parent_actor):
#         self.sensor = None
#         self._parent = parent_actor
#         bound_x = 0.5 + self._parent.bounding_box.extent.x
#         bound_y = 0.5 + self._parent.bounding_box.extent.y
#         bound_z = 0.5 + self._parent.bounding_box.extent.z
#
#         self.velocity_range = 7.5 # m/s
#         world = self._parent.get_world()
#         self.debug = world.debug
#         bp = world.get_blueprint_library().find('sensor.other.radar')
#         bp.set_attribute('horizontal_fov', str(35))
#         bp.set_attribute('vertical_fov', str(20))
#         self.sensor = world.spawn_actor(
#             bp,
#             carla.Transform(
#                 carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
#                 carla.Rotation(pitch=5)),
#             attach_to=self._parent)
#         # We need a weak reference to self to avoid circular reference.
#         weak_self = weakref.ref(self)
#         self.sensor.listen(
#             lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))
#
#     @staticmethod
#     def _Radar_callback(weak_self, radar_data):
#         print(f"Radar detected {len(radar_data)} points")
#         self = weak_self()
#         if not self:
#             return
#         # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
#         # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
#         # points = np.reshape(points, (len(radar_data), 4))
#
#         current_rot = radar_data.transform.rotation
#         for detect in radar_data:
#             azi = math.degrees(detect.azimuth)
#             alt = math.degrees(detect.altitude)
#             # The 0.25 adjusts a bit the distance so the dots can
#             # be properly seen
#             fw_vec = carla.Vector3D(x=detect.depth - 0.25)
#             carla.Transform(
#                 carla.Location(),
#                 carla.Rotation(
#                     pitch=current_rot.pitch + alt,
#                     yaw=current_rot.yaw + azi,
#                     roll=current_rot.roll)).transform(fw_vec)
#
#             def clamp(min_v, max_v, value):
#                 return max(min_v, min(value, max_v))
#
#             norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
#             r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
#             g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
#             b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
#             self.debug.draw_point(
#                 radar_data.transform.location + fw_vec,
#                 size=0.05,
#                 life_time=1.0,
#                 persistent_lines=False,
#                 color=carla.Color(r, g, b))

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
        self.create_ego_cameras()
        # Set spectator
        self.spectator = self.world.get_spectator()

        self.add_radar()

    def tick(self):
        self.world.tick()
        # Update spectator view
        self.update_spectator()

    def create_world(self):
        self.client = carla.Client('localhost', self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
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
        for i in range(0, 25):
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

    def create_ego_cameras(self):
        camera_init_trans = carla.Transform(carla.Location(z=1.5,x=1.5), carla.Rotation(pitch=0, yaw=0, roll=0))
        # We create the camera through a blueprint that defines its properties
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "480")
        camera_bp.set_attribute("sensor_tick", "0.05")
        # We spawn the camera and attach it to our ego vehicle
        self.rgb_camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.ego_vehicle)
        self.rgb_camera_queue = queue.Queue(maxsize=10)
        self.rgb_camera.listen(lambda image: self.rgb_camera_queue.put_nowait(image))

        # Depth camera setup
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
        # Compute position 10m behind and 5m above ego car
        forward_vector = transform.get_forward_vector()
        spectator_location = transform.location - 10 * forward_vector + carla.Location(z=5)
        spectator_transform = carla.Transform(spectator_location, transform.rotation)
        self.spectator.set_transform(spectator_transform)

    def add_radar(self):
        self.radar_sensor=RadarSensor(self.ego_vehicle,self.rgb_camera)

    def expose_queues(self):
        return self.rgb_camera_queue, self.depth_camera_queue

    def cleanup(self):
        self.rgb_camera.stop()
        self.rgb_camera.destroy()
        self.depth_camera.stop()
        self.depth_camera.destroy()
        self.ego_vehicle.destroy()