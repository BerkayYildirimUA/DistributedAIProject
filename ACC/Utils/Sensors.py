import random
import time
import weakref
import collections

import carla
from IPython.core.inputtransformer2 import leading_empty_lines
from carla import Vector3D
import math
import logging

from jedi.debug import speed

from ACC.Engine.engine import SingletonLightState
from ACC.Utils.GForce_Class import GForceCalculator
from ACC.Utils.abstractions import StateSensor, UI, VehicleState, LightColors

class CarlaWorldStateSensor(StateSensor):

    def __init__(self, ego_vehicle: carla.Vehicle, world: carla.World):
        self.__ego = ego_vehicle
        self.__world = world
        self.__map = self.__world.get_map()

        self.__safe_time_distance_seconds = 3
        self.counter = 0
        self.__collision_sensor = CollisionSensor(ego_vehicle)

        self.override_speed_limit = False
        self.speed_limit = 0

        self.__g_force_ego_calculator = GForceCalculator(self.__world.get_settings().fixed_delta_seconds)
        self.__relative_speed_lead_calculator = GForceCalculator(self.__world.get_settings().fixed_delta_seconds)
        self.__speed_light_calculator = GForceCalculator(self.__world.get_settings().fixed_delta_seconds)

        self.min_dist = 250

        self._cached_traffic_lights = None
        self._traffic_light_cache_frame = -1  # Track when cache was created


    def cleanup(self):

        if self.__collision_sensor is not None:
            try:
                self.__collision_sensor.destroy()
            except Exception as e:
                logging.warning(f"Error destroying collision sensor: {e}")
            self.__collision_sensor = None

        self.__ego = None
        self.__world = None
        self.__map = None
        self._cached_traffic_lights = None

    def _get_light_color_enum(self, carla_state):
        """Maps CARLA TrafficLightState to LightColors Enum"""

        temp = SingletonLightState().get_state()
        if temp != "OFF":
            carla_state = temp

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

            current_frame = self.__world.get_snapshot().frame if self.__world else 0
            if (self._cached_traffic_lights is None or
                    current_frame - self._traffic_light_cache_frame > 5):
                try:
                    self._cached_traffic_lights = list(self.__world.get_actors().filter('traffic.traffic_light'))
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
                ego_extent = self.__ego.bounding_box.extent.x
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
        if not self.__ego:
            return self._get_default_crashed_state()

        try:
            if not self.__ego.is_alive:
                return self._get_default_crashed_state()
        except RuntimeError:
            return self._get_default_crashed_state()

        try:
            ego_transform = self.__ego.get_transform()
            ego_loc = ego_transform.location
            ego_forward = ego_transform.get_forward_vector()
        except RuntimeError:
            return self._get_default_crashed_state()

        # Safely get all vehicles
        try:
            all_vehicles = self.__world.get_actors().filter('vehicle.*')
        except RuntimeError:
            all_vehicles = []

        # Build list with safe access
        vehicle_data = []
        for v in all_vehicles:
            if v.id != self.__ego.id:
                data = self._safe_get_vehicle_data(v, ego_loc)
                if data:
                    vehicle_data.append(data)

        # Get ego velocity safely
        try:
            ego_velocity_vec: Vector3D = self.__ego.get_velocity()
            ego_velocity_ms = ego_velocity_vec.length()
        except RuntimeError:
            ego_velocity_ms = 0.0

        safe_distance_m = self.__safe_time_distance_seconds * ego_velocity_ms

        # Check collision safely
        last_impact = False
        if self.__collision_sensor:
            try:
                last_impact = self.__collision_sensor.get_last_impact()
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
                self.speed_limit = self.__ego.get_speed_limit()
            except RuntimeError:
                pass

        if self.speed_limit == 0.0:
            self.speed_limit = 30

        # Traffic lights
        traffic_light_dist_m = self.min_dist
        traffic_light_color = LightColors.green

        try:
            ego_waypoint = self.__map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
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

        if self.counter % 300 == 0 or last_impact > 0.0:
            logging.info(f"speed: {ego_velocity_ms * 3.6}km/h, speed lim: {self.speed_limit} km/h, "
                         f"distance to nearest: {smallest_dist}m, safe dist: {safe_distance_m}m, CRASH: {last_impact}")

        self.counter += 1

        # Update calculators
        self.__g_force_ego_calculator.update_speed(ego_velocity_ms)
        self.__relative_speed_lead_calculator.update_speed(result_dist_m)
        self.__speed_light_calculator.update_speed(traffic_light_dist_m)

        ego_g_force = self.__g_force_ego_calculator.get_latest_g_force() or 0
        relative_speed_ms = self.__relative_speed_lead_calculator.get_latest_g_force() or 0
        speed_light_ms = self.__speed_light_calculator.get_latest_g_force() or 0

        return VehicleState(
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