import queue
import random
import threading
import time
import weakref

import carla
import cv2
from carla import Vector3D
import math
import logging
import numpy as np
from typing_extensions import override

from ACC.Utils.GForce_Class import GForceCalculator
from ACC.Utils.abstractions import StateSensor, UI, VehicleState, LightColors
import app.constants  as constants
from app.memory.shared_memory import RGBCameraMemory, VehicleDistanceMemory, RadarMemory, CameraCalibrationMemory


class CarlaWorldStateSensor(StateSensor):

    def __init__(self, ego_vehicle: carla.Vehicle, world: carla.World):
        self._ego = ego_vehicle
        self._world = world
        self._map = self._world.get_map()

        self._safe_time_distance_seconds = 3
        self.counter = 0
        self._collision_sensor = CollisionSensor(ego_vehicle)

        self.override_speed_limit = False
        self.speed_limit = 0

        self._g_force_ego_calculator = GForceCalculator(self._world.get_settings().fixed_delta_seconds)
        self._relative_speed_lead_calculator = GForceCalculator(self._world.get_settings().fixed_delta_seconds)
        self._speed_light_calculator = GForceCalculator(self._world.get_settings().fixed_delta_seconds)

        self.min_dist = 250

        self._cached_traffic_lights = None
        self._traffic_light_cache_frame = -1  # Track when cache was created



    def log_vehicle_state(self,state:VehicleState):
        logging.info(
            f"speed: {state.speed_ms * 3.6:.2f}fkm/h, speed limit: {state.speed_limit_ms*3.6:.2f} km/h, distance to nearest: {state.lead_distance_m:.2f}m, safe dist: {state.safe_following_distance_m:.2f}m,traffic light color: {state.light_color}, traffic light distance: {state.light_dist_m:.2f}m, CRASH: {state.crash_intensity}, g-forces:{state.g_force_ego}, {state.relative_speed_ms}, {state.light_speed_ms}")

    def cleanup(self):

        if self._collision_sensor is not None:
            try:
                self._collision_sensor.destroy()
            except Exception as e:
                logging.warning(f"Error destroying collision sensor: {e}")
            self._collision_sensor = None

        self._ego = None
        self._world = None
        self._map = None
        self._cached_traffic_lights = None

    def _get_light_color_enum(self, carla_state):
        """Maps CARLA TrafficLightState to LightColors Enum"""
        if carla_state == carla.TrafficLightState.Red:
            return LightColors.red
        elif carla_state == carla.TrafficLightState.Yellow:
            return LightColors.orange
        else:
            return LightColors.green


    def _get_trafficlight(self, ego_waypoint):
        landmarks = ego_waypoint.get_landmarks_of_type(600.0, "1000001", stop_at_junction=False)

        target_light_actor = None


        if landmarks:
            target_landmark = landmarks[0]
            landmark_loc = target_landmark.transform.location

            current_frame = self._world.get_snapshot().frame if self._world else 0
            if (self._cached_traffic_lights is None or
                    current_frame - self._traffic_light_cache_frame > 5):
                try:
                    self._cached_traffic_lights = list(self._world.get_actors().filter('traffic.traffic_light'))
                    self._traffic_light_cache_frame = current_frame
                except Exception:
                    self._cached_traffic_lights = []


            closest_dist = float('inf')

            for tl_actor in self._cached_traffic_lights:
                try:
                    if not tl_actor.is_alive:
                        continue

                    dist = tl_actor.get_location().distance(landmark_loc)

                    if dist < 2.0:
                        target_light_actor = tl_actor
                        break

                    if dist < closest_dist:
                        closest_dist = dist
                except RuntimeError:
                    # Actor was destroyed between check and access
                    continue

        return target_light_actor

    def _safe_get_vehicle_data(self, vehicle, ego_loc):
        """Safely get vehicle distance (BUMPER-TO-BUMPER) and transform"""
        try:
            if not vehicle.is_alive:
                return None

            other_loc = vehicle.get_location()

            # Center-to-center distance
            center_dist = math.sqrt(
                (other_loc.x - ego_loc.x) ** 2 +
                (other_loc.y - ego_loc.y) ** 2 +
                (other_loc.z - ego_loc.z) ** 2
            )

            # ADJUST FOR BOUNDING BOXES (bumper-to-bumper)
            try:
                ego_extent = self._ego.bounding_box.extent.x
                other_extent = vehicle.bounding_box.extent.x
                bumper_dist = max(0.0, center_dist - ego_extent - other_extent)
            except Exception:
                bumper_dist = max(0.0, center_dist - 6)

            transform = vehicle.get_transform()
            return (bumper_dist, vehicle, transform)

        except RuntimeError:
            return None

    def get_state(self) -> VehicleState:
        # Early exit if ego is invalid
        if not self._ego:
            return self._get_default_crashed_state()

        try:
            if not self._ego.is_alive:
                return self._get_default_crashed_state()
        except RuntimeError:
            return self._get_default_crashed_state()

        try:
            ego_transform = self._ego.get_transform()
            ego_loc = ego_transform.location
            ego_forward = ego_transform.get_forward_vector()
        except RuntimeError:
            return self._get_default_crashed_state()

        # Safely get all vehicles
        try:
            all_vehicles = self._world.get_actors().filter('vehicle.*')
        except RuntimeError:
            all_vehicles = []

        # Build list with safe access
        vehicle_data = []
        for v in all_vehicles:
            if v.id != self._ego.id:
                data = self._safe_get_vehicle_data(v, ego_loc)
                if data:
                    vehicle_data.append(data)

        # Get ego velocity safely
        try:
            ego_velocity_vec: Vector3D = self._ego.get_velocity()
            ego_velocity_ms = ego_velocity_vec.length()
        except RuntimeError:
            ego_velocity_ms = 0.0

        safe_distance_m = self._safe_time_distance_seconds * ego_velocity_ms

        # Check collision safely
        last_impact = False
        if self._collision_sensor:
            try:
                last_impact = self._collision_sensor.get_last_impact()
            except Exception:
                pass

        # Find nearest vehicle in front
        smallest_dist = self.min_dist
        for dist, vehicle, v_transform in sorted(vehicle_data, key=lambda x: x[0]):
            try:
                dot = v_transform.get_forward_vector().dot(ego_forward)
                if smallest_dist > dist and dot > 0.8:
                    smallest_dist = dist
            except RuntimeError:
                continue

        result_dist_m = smallest_dist

        # Speed limit handling
        if self.override_speed_limit and self.counter == 2500:
            self.counter = 0
            self.speed_limit = random.randint(10, 140)
        elif not self.override_speed_limit:
            try:
                self.speed_limit = self._ego.get_speed_limit()
            except RuntimeError:
                pass

        if self.speed_limit == 0.0:
            self.speed_limit = 30

        # Traffic lights
        traffic_light_dist_m = self.min_dist
        traffic_light_color = LightColors.green

        try:
            ego_waypoint = self._map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            target_light_actor = self._get_trafficlight(ego_waypoint)

            if target_light_actor:
                try:
                    light_loc : carla.Location = target_light_actor.get_location()
                    road_vec = ego_waypoint.transform.get_forward_vector()

                    car_to_light_vec = light_loc - ego_loc
                    longitudinal_dist = (car_to_light_vec.x * road_vec.x) + (car_to_light_vec.y * road_vec.y)

                    if longitudinal_dist > 0:
                        traffic_light_dist_m = min(longitudinal_dist, traffic_light_dist_m)
                        traffic_light_color = self._get_light_color_enum(target_light_actor.get_state())

                except RuntimeError:
                    pass
        except Exception:
            pass

        self.counter += 1

        # Update calculators
        self._g_force_ego_calculator.update_speed(ego_velocity_ms)
        self._relative_speed_lead_calculator.update_speed(result_dist_m)
        self._speed_light_calculator.update_speed(traffic_light_dist_m)

        ego_g_force = self._g_force_ego_calculator.get_latest_g_force() or 0
        relative_speed_ms = self._relative_speed_lead_calculator.get_latest_g_force() or 0
        speed_light_ms = self._speed_light_calculator.get_latest_g_force() or 0
        
        state = VehicleState(
            speed_ms=ego_velocity_ms,
            speed_limit_ms=self.speed_limit / 3.6,
            lead_distance_m=result_dist_m,
            safe_following_distance_m=safe_distance_m,
            crash_intensity=last_impact,
            light_color=traffic_light_color,
            light_dist_m=traffic_light_dist_m,
            g_force_ego=ego_g_force,
            relative_speed_ms=relative_speed_ms,
            light_speed_ms=speed_light_ms
        )
        if self.counter % 300 == 0 or last_impact > 0.0:
            self.log_vehicle_state(state)
        return state

    def _get_default_crashed_state(self):
        """Return a safe default state when ego is invalid"""
        return VehicleState(
            speed_ms=0,
            speed_limit_ms=30 / 3.6,
            lead_distance_m=self.min_dist,
            safe_following_distance_m=10,
            crash_intensity=1,
            light_color=LightColors.green,
            light_dist_m=self.min_dist,
            g_force_ego=0,
            relative_speed_ms=0,
            light_speed_ms=0
        )

class CarlaVBWorldStateSensor(CarlaWorldStateSensor):

    def __init__(self, ego_vehicle: carla.Vehicle, world: carla.World):
        super().__init__(ego_vehicle, world)

        # self._safe_time_distance_seconds = 3
        # self.counter = 0
        #
        # self.override_speed_limit = False
        # self.speed_limit = 0

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

        self.prev_lead_distance=250.0

    def cleanup(self):
        pass

    def reset(self, ego, world):

        self._ego = ego
        self._world = world
        self.counter = 0

    @override
    def get_state(self) -> VehicleState:
        ego_velocity_vec: Vector3D = self._ego.get_velocity()
        ego_velocity_ms = ego_velocity_vec.length()

        safe_distance = self._safe_time_distance_seconds * ego_velocity_ms

        self.speed_limit = self._ego.get_speed_limit()

        distance= self.vehicle_distance_memory.read()
        # Keep track of previous distance and use it in case radar returns inf values
        if np.isinf(distance[0]):
            lead_distance=self.prev_lead_distance
        else:
            self.prev_lead_distance = distance[0]
            lead_distance = distance[0]

        # TODO: replace with computer vision based logic
        traffic_light_dist_m = self.min_dist
        traffic_light_color = LightColors.green

        # TODO: get speed limit from computer vision
        speed_limit=self.speed_limit

        # G-force
        self._g_force_ego_calculator.update_speed(ego_velocity_ms)
        self._relative_speed_lead_calculator.update_speed(distance[0])
        self._speed_light_calculator.update_speed(traffic_light_dist_m)

        ego_g_force = self._g_force_ego_calculator.get_latest_g_force() or 0
        relative_speed_ms = self._relative_speed_lead_calculator.get_latest_g_force() or 0
        speed_light_ms = self._speed_light_calculator.get_latest_g_force() or 0


        self.counter += 1

        ctrl = self._ego.get_control()  # get the control applied in the last tick
        # ctrl.steer in [-1,1] => schaal naar rad
        steer_rad = -float(ctrl.steer) * constants.MAX_STEER_RAD
        state2 = super().get_state()

        state= VehicleState(
            speed_ms=ego_velocity_ms,
            speed_limit_ms=speed_limit/3.6,
            lead_distance_m=lead_distance,
            safe_following_distance_m=safe_distance,
            crash_intensity=0.0,
            light_color=traffic_light_color,
            light_dist_m=traffic_light_dist_m,
            g_force_ego=ego_g_force,
            relative_speed_ms=relative_speed_ms,
            light_speed_ms=speed_light_ms,
            steer_rad=steer_rad
        )
        if self.counter % 300 == 0:
            print("Grount truth")
            self.log_vehicle_state(state2)
            print("Vision based")
            self.log_vehicle_state(state)
        return state2


    def create_ego_sensors(self):
        sensor_location = carla.Location(x=constants.SENSOR_POS_X, z=constants.SENSOR_POS_Z)
        sensor_rotation = carla.Rotation(pitch=constants.SENSOR_PITCH, yaw=constants.SENSOR_YAW,
                                         roll=constants.SENSOR_ROLL)
        camera_init_trans = carla.Transform(sensor_location, sensor_rotation)

        # We create the camera through a blueprint that defines its properties
        camera_bp = self._world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", str(constants.IMAGE_WIDTH))
        camera_bp.set_attribute("image_size_y", str(constants.IMAGE_HEIGHT))
        camera_bp.set_attribute("sensor_tick", str(constants.SENSOR_TICK))
        camera_bp.set_attribute("fov", str(constants.HOR_FOV_DEG))
        # We spawn the camera and attach it to our ego vehicle
        self.rgb_camera = self._world.spawn_actor(camera_bp, camera_init_trans,
                                                                  attach_to=self._ego)
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
        blueprint_library = self._world.get_blueprint_library()
        radar_bp = blueprint_library.find('sensor.other.radar')
        # TODO: change these parameters to values found in real radar setups
        radar_bp.set_attribute('horizontal_fov', str(constants.HOR_FOV_DEG))
        radar_bp.set_attribute('vertical_fov', str(constants.VERT_FOV_DEG))
        radar_bp.set_attribute('range', str(constants.RADAR_RANGE))
        radar_bp.set_attribute('points_per_second', '30000')
        radar_bp.set_attribute('sensor_tick', str(constants.SENSOR_TICK))
        radar_transform = carla.Transform(sensor_location, sensor_rotation)
        self.radar = self._world.spawn_actor(radar_bp, radar_transform, attach_to=self._ego)
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

#code form carla examples, from "automatic_control.py"
class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.intensity = 0.0  # Store the intensity of the last impact
        self._destroyed = False
        self._callback_lock = False

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
        if not self or self._destroyed or self._callback_lock:
            return

        try:
            impulse = event.normal_impulse
            intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)

            self.history.append((event.frame, intensity))
            self.intensity = intensity

            if len(self.history) > 4000:
                self.history.pop(0)
        except Exception:
            pass

    def get_last_impact(self):
        """Returns the intensity of the most recent collision, then resets it."""
        current_intensity = self.intensity
        self.intensity = 0.0  # Reset after reading so we don't register the same crash twice
        return current_intensity

    def destroy(self):
        """Clean up the sensor from the server - improved version"""
        if self._destroyed:
            return

        self._destroyed = True
        self._callback_lock = True  # Block any pending callbacks

        if self.sensor is not None:
            try:
                # Stop listening first
                if self.sensor.is_listening:
                    self.sensor.stop()

                # Small delay to let pending callbacks drain
                time.sleep(0.1)

                # Now destroy
                if self.sensor.is_alive:
                    self.sensor.destroy()
            except Exception as e:
                logging.debug(f"Error during collision sensor cleanup: {e}")
            finally:
                self.sensor = None

        self._parent = None
        self.history.clear()


class PygameUI(UI):
    pass