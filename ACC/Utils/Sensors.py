import queue
import random
import threading
import weakref

import carla
import cv2
from carla import Vector3D
import math
import logging
import numpy as np

from ACC.Utils.abstractions import StateSensor, UI, VehicleState, LightColors
import app.constants  as constants
from app.memory.shared_memory import RGBCameraMemory, VehicleDistanceMemory, RadarMemory, CameraCalibrationMemory


class CarlaWorldStateSensor(StateSensor):

    def __init__(self, ego_vehicle: carla.Vehicle, world: carla.World):
        self.__ego = ego_vehicle
        self.__world = world

        self.__safe_time_distance_seconds = 2
        self.counter = 0
        self.__collision_sensor = CollisionSensor(ego_vehicle)

        self.override_speed_limit = False
        self.speed_limit = 0

    def cleanup(self):

        if self.__collision_sensor:
            self.__collision_sensor.destroy()

    def reset(self, ego, world):

        self.__ego = ego
        self.__world = world
        self.counter = 0
        if self.__collision_sensor:
            self.__collision_sensor.destroy()



    def get_state(self) -> VehicleState:

        ego_transform = self.__ego.get_transform()

        vehicles = self.__world.get_actors().filter('vehicle.*')

        dist = lambda l : math.sqrt((l.x - ego_transform.location.x)**2 + (l.y - ego_transform.location.y)
                             ** 2 + (l.z - ego_transform.location.z)**2)

        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != self.__ego.id]


        ego_velocity_vec: Vector3D = self.__ego.get_velocity()
        ego_velocity_ms = ego_velocity_vec.length()

        ctrl = self.__ego.get_control()  # get the control applied in the last tick
        # ctrl.steer in [-1,1] => schaal naar rad
        steer_rad = -float(ctrl.steer) * constants.MAX_STEER_RAD

        safe_distance = self.__safe_time_distance_seconds * ego_velocity_ms
        has_crashed = self.__collision_sensor.get_last_impact() > 0.0

        smallest_dist = 400
        dists = []
        for dist, vehicle in sorted(vehicles):
            dot = vehicle.get_transform().get_forward_vector().dot(ego_transform.get_forward_vector()) # to see if the car is pointing the same way as the ego
            if smallest_dist > dist and dot > 0.8:
                smallest_dist = dist
            dists.append(dist)

        if len(vehicles) == 0:
            dists.append(400)


        if self.override_speed_limit and self.counter == 5000:
            self.counter = 0
            self.speed_limit = random.randint(10, 140)
        elif not self.override_speed_limit:
            self.speed_limit = self.__ego.get_speed_limit()

        if self.speed_limit == 0.0:
            self.speed_limit = 30


        if self.counter % 300 == 0:
            logging.info(f"speed: {ego_velocity_ms * 3.6}km/h, speed lim: {self.speed_limit} km/h, distance to nearest: {smallest_dist}m, safe dist: {safe_distance}m, CRASH: {has_crashed}")

        self.counter += 1


        return VehicleState(speed=ego_velocity_ms * 3.6, speed_limit=self.speed_limit, distances=dists, safe_following_distance=safe_distance, hasCrashed=has_crashed, light_color=LightColors.green,steering_dir=steer_rad)

class CarlaVBWorldStateSensor(StateSensor):

    def __init__(self, ego_vehicle: carla.Vehicle, world: carla.World):
        self.__ego = ego_vehicle
        self.__world = world

        self.__safe_time_distance_seconds = 2
        self.counter = 0

        self.override_speed_limit = False
        self.speed_limit = 0

        # Create Sensors
        self.create_ego_sensors()

        # Create sensor memories
        self.rgb_camera_memory = RGBCameraMemory().get_write_access()
        # depht_camera_memory = DepthCameraMemory().get_write_access()
        self.vehicle_distance_memory = VehicleDistanceMemory().get_read_access()
        self.radar_memory = RadarMemory().get_write_access()
        self.camera_calibration_memory = CameraCalibrationMemory().get_write_access()
        # Create camera properties
        K = self.calculate_camera_intrinsic()
        self.cam_mats = np.zeros((2, 4, 4), dtype=np.float64)
        self.cam_mats[0, :3, :3] = K  # intrinsic (3x3 in top-left corner)

        self.start_sensor_threads()

    def cleanup(self):
        pass

    def reset(self, ego, world):

        self.__ego = ego
        self.__world = world
        self.counter = 0

    def get_state(self) -> VehicleState:
        ego_velocity_vec: Vector3D = self.__ego.get_velocity()
        ego_velocity_ms = ego_velocity_vec.length()

        ctrl = self.__ego.get_control()  # get the control applied in the last tick
        # ctrl.steer in [-1,1] => schaal naar rad
        steer_rad = -float(ctrl.steer) * constants.MAX_STEER_RAD

        safe_distance = self.__safe_time_distance_seconds * ego_velocity_ms

        self.speed_limit = self.__ego.get_speed_limit()

        distance= self.vehicle_distance_memory.read()

        if self.counter % 300 == 0:
            logging.info(f"speed: {ego_velocity_ms * 3.6}km/h, speed lim: {self.speed_limit} km/h, distance to nearest: {distance}m, safe dist: {safe_distance}m, CRASH: nvt")

        self.counter += 1
        # TODO add traffic light logic
        return VehicleState(speed=ego_velocity_ms * 3.6, speed_limit=self.speed_limit, distances=distance, safe_following_distance=safe_distance, hasCrashed=False, light_color=LightColors.green,steering_dir=steer_rad)

    def create_ego_sensors(self):
        sensor_location = carla.Location(x=constants.SENSOR_POS_X, z=constants.SENSOR_POS_Z)
        sensor_rotation = carla.Rotation(pitch=constants.SENSOR_PITCH, yaw=constants.SENSOR_YAW,
                                         roll=constants.SENSOR_ROLL)
        camera_init_trans = carla.Transform(sensor_location, sensor_rotation)

        # We create the camera through a blueprint that defines its properties
        camera_bp = self.__world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", str(constants.IMAGE_WIDTH))
        camera_bp.set_attribute("image_size_y", str(constants.IMAGE_HEIGHT))
        camera_bp.set_attribute("sensor_tick", str(constants.SENSOR_TICK))
        camera_bp.set_attribute("fov", str(constants.HOR_FOV_DEG))
        # We spawn the camera and attach it to our ego vehicle
        self.rgb_camera = self.__world.spawn_actor(camera_bp, camera_init_trans,
                                                                      attach_to=self.__ego)
        self.rgb_camera_queue = queue.Queue(maxsize=constants.QUEUE_MAXSIZE)
        # self.rgb_camera.listen(lambda image: self.rgb_camera_queue.put_nowait(image))
        self.rgb_camera.listen(lambda data: (self.rgb_camera_queue.get_nowait(), self.rgb_camera_queue.put_nowait(
            data)) if self.rgb_camera_queue.full() else self.rgb_camera_queue.put_nowait(data))

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
        blueprint_library = self.__world.get_blueprint_library()
        radar_bp = blueprint_library.find('sensor.other.radar')
        # TODO: change these parameters to values found in real radar setups
        radar_bp.set_attribute('horizontal_fov', str(constants.HOR_FOV_DEG))
        radar_bp.set_attribute('vertical_fov', str(constants.VERT_FOV_DEG))
        radar_bp.set_attribute('range', str(constants.RADAR_RANGE))
        radar_bp.set_attribute('points_per_second', '30000')
        radar_bp.set_attribute('sensor_tick', str(constants.SENSOR_TICK))
        radar_transform = carla.Transform(sensor_location, sensor_rotation)
        self.radar = self.__world.spawn_actor(radar_bp, radar_transform, attach_to=self.__ego)
        self.radar_queue = queue.Queue(maxsize=constants.QUEUE_MAXSIZE)
        # check if queue is full: yes --> pop oldest, push new one. no --> push. Ensures most recent radar data is in the queue
        self.radar.listen(lambda data: (self.radar_queue.get_nowait(), self.radar_queue.put_nowait(
            data)) if self.radar_queue.full() else self.radar_queue.put_nowait(data))

        print("Camera attrs:", self.rgb_camera.attributes)
        print("Radar attrs:", self.radar.attributes)


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

    def process_rgb_images(self):
        while True:
            try:
                image = self.rgb_camera_queue.get(timeout=1.0)
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))
                new_frame = array[:, :, :3]
                frame_send_to_inference = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
                self.rgb_camera_memory.write(frame_send_to_inference)
            except queue.Empty:
                continue


    def process_radar_data(self):
        while True:
            try:
                radar_data = self.radar_queue.get(timeout=1.0)
                # Radar callback (manual_control.py logic from PythonAPI/examples)
                max_n = constants.RADAR_MAX_DETECTIONS

                # We will always write a (max_n, 5) array: [x, y, z, depth, velocity]
                # Zero rows mean "padding" and will be ignored, used for consistency in shared memory
                points = np.zeros((max_n, 5), dtype=np.float32)

                n = len(radar_data)
                if n == 0:
                    self.radar_memory.write(points)  # all zeros
                    return

                current_rot = radar_data.transform.rotation  # sensor rotation (world)
                sensor_loc = radar_data.transform.location  # sensor location (world)

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
                            yaw=current_rot.yaw + azi,
                            roll=current_rot.roll)
                    ).transform(fw_vec)

                    # world point = sensor location + rotated forward vector
                    world_point = sensor_loc + fw_vec

                    points[i, 0] = world_point.x
                    points[i, 1] = world_point.y
                    points[i, 2] = world_point.z
                    points[i, 3] = det.depth
                    points[i, 4] = det.velocity

                self.radar_memory.write(points)

                P = self.calculate_camera_extrinsic()
                self.cam_mats[1, :, :] = P  # extrinsic, full 4x4 world -> camera (cv frame)
                self.camera_calibration_memory.write(self.cam_mats)

            except queue.Empty:
                continue

    def start_sensor_threads(self):
        # Start threads
        rgb_thread = threading.Thread(target=self.process_rgb_images, daemon=True)
        # depth_thread = threading.Thread(target=self.process_depth_images, daemon=True)
        radar_thread = threading.Thread(target=self.process_radar_data, daemon=True)
        rgb_thread.start()
        # depth_thread.start()
        radar_thread.start()


class CarlaLeadStateSensor(StateSensor):

    def __init__(self, ego_vehicle: carla.Actor, lead: carla.Actor = None):
        self.__ego = ego_vehicle
        self.__lead = lead

        self.__safe_time_distance_seconds = 2
        self.counter = 0
        self.__collision_sensor = CollisionSensor(ego_vehicle)


    def get_state(self) -> VehicleState:
        ego_transform = self.__ego.get_transform()
        lead_transform = self.__lead.get_transform()

        distance = ego_transform.location.distance(lead_transform.location)

        ego_velocity_vec: Vector3D = self.__ego.get_velocity()
        ego_velocity_ms = ego_velocity_vec.length()

        safe_distance = self.__safe_time_distance_seconds * ego_velocity_ms

        has_crashed = self.__collision_sensor.get_last_impact() > 0.0

        speed_limit = self.__ego.get_speed_limit()

        if speed_limit == 0.0:
            speed_limit = 30

        if self.counter == 1000:
            self.counter = 0
            logging.info(f"speed: {ego_velocity_ms * 3.6}km/h, speed lim: {speed_limit} km/h, distance to nearest: {distance}m, safe dist: {safe_distance}m")
        else:
            self.counter += 1

        speed_limit = self.__ego.get_speed_limit()

        return VehicleState(speed=ego_velocity_ms * 3.6, speed_limit=speed_limit, distances=[distance], safe_following_distance=safe_distance, hasCrashed=has_crashed, light_color=LightColors.green)


#code form carla examples, from "automatic_control.py"
class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.intensity = 0.0  # Store the intensity of the last impact

        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')

        # Spawn the sensor attached to the parent
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

        # We use weakref to avoid circular references that prevent garbage collection
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Calculate intensity of the collision
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)

        # Append to history (Frame number, Intensity)
        self.history.append((event.frame, intensity))

        # Update current intensity state
        self.intensity = intensity

        # Keep history manageable (optional limit)
        if len(self.history) > 4000:
            self.history.pop(0)

    def get_last_impact(self):
        """Returns the intensity of the most recent collision, then resets it."""
        current_intensity = self.intensity
        self.intensity = 0.0  # Reset after reading so we don't register the same crash twice
        return current_intensity

    def destroy(self):
        """Clean up the sensor from the server"""
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None

class PygameUI(UI):
    pass