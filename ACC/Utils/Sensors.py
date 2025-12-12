import random
import time
import weakref

import carla
from carla import Vector3D
import math
import logging
import keyboard

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
        elif carla_state == carla.TrafficLightState.Green:
            return LightColors.green
        return None

    def _light_up_actor_box(self, actor : carla.TrafficLight):

        box = actor.bounding_box
        actor_transform = actor.get_transform()

        box.location = actor_transform.transform(box.location)

        r = 1 if actor.get_state() == carla.TrafficLightState.Red else 0
        g = 1 if actor.get_state() == carla.TrafficLightState.Green else 0
        y = 1 if actor.get_state() == carla.TrafficLightState.Yellow else 0


        # 2. Draw the box using the debug helper.
        # We use a Green color (0, 255, 0) and thickness to create the 'glow' effect.
        self.__world.debug.draw_box(
            box=box,
            rotation=actor_transform.rotation,  # Orient the box to match the actor
            thickness=0.1,  # Thicker lines for a "glow" effect
            color=carla.Color(r, g,  y),  # Christmas Green
            life_time=0.1  # Short lifetime for continuous updates
        )

    def _get_trafficlight_simple(self, ego_loc, ego_forward):
        """Find traffic light using CARLA's built-in methods + spatial search"""

        # Method 1: CARLA's built-in (works when close to junction)
        try:
            if self.__ego.is_at_traffic_light():
                light = self.__ego.get_traffic_light()
                if light and light.is_alive:
                    return light
        except Exception:
            pass

        # Method 2: Spatial search for lights ahead of us
        current_frame = self.__world.get_snapshot().frame if self.__world else 0
        if (self._cached_traffic_lights is None or
                current_frame - self._traffic_light_cache_frame > 5):
            try:
                self._cached_traffic_lights = list(self.__world.get_actors().filter('traffic.traffic_light'))
                self._traffic_light_cache_frame = current_frame
            except Exception:
                self._cached_traffic_lights = []

        best_light = None
        best_dist = self.min_dist

        for tl_actor in self._cached_traffic_lights:
            try:
                if not tl_actor.is_alive:
                    continue

                # Get BB and transform center to world space
                bb = tl_actor.bounding_box
                actor_transform = tl_actor.get_transform()
                bb_center = actor_transform.transform(bb.location)

                # Vector from ego to BB center
                to_light_x = bb_center.x - ego_loc.x
                to_light_y = bb_center.y - ego_loc.y

                # Longitudinal distance
                longitudinal = to_light_x * ego_forward.x + to_light_y * ego_forward.y

                if longitudinal < 0 or longitudinal > self.min_dist:
                    continue

                # BB's x-axis direction in world space
                total_yaw = math.radians(actor_transform.rotation.yaw + bb.rotation.yaw)
                bb_x_dir_x = math.cos(total_yaw)
                bb_x_dir_y = math.sin(total_yaw)

                # Number of segments = extent.x (3.0 → 3 segments)
                extent_x = bb.extent.x
                num_segments = max(1, int(extent_x))

                # Check each segment
                in_lane = False
                for i in range(num_segments):
                    # Spread points from -extent_x to +extent_x
                    if num_segments == 1:
                        offset = 0.0
                    else:
                        offset = -extent_x + (2.0 * extent_x) * i / (num_segments - 1)

                    # World position of this segment
                    segment_x = bb_center.x + bb_x_dir_x * offset
                    segment_y = bb_center.y + bb_x_dir_y * offset

                    # Lateral offset from ego's path
                    to_seg_x = segment_x - ego_loc.x
                    to_seg_y = segment_y - ego_loc.y
                    lateral = abs(-ego_forward.y * to_seg_x + ego_forward.x * to_seg_y)

                    if lateral < 3.0:
                        in_lane = True
                        break

                if not in_lane:
                    continue

                # Steering check
                angle = self.__ego.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)

                if longitudinal < best_dist and abs(angle) < 3:
                    best_dist = longitudinal
                    best_light = tl_actor

            except RuntimeError:
                continue

        return best_light

    def _safe_get_vehicle_data(self, vehicle, ego_loc, ego_forward):
        """Safely get vehicle distance (BUMPER-TO-BUMPER) and check if in front"""
        try:
            if not vehicle.is_alive:
                return None

            other_loc = vehicle.get_location()

            # Vector from ego to other vehicle
            to_other_x = other_loc.x - ego_loc.x
            to_other_y = other_loc.y - ego_loc.y
            to_other_z = other_loc.z - ego_loc.z

            center_dist = math.sqrt(to_other_x ** 2 + to_other_y ** 2 + to_other_z ** 2)

            if center_dist < 0.01:
                return None

            # Normalize direction to other vehicle
            to_other_norm_x = to_other_x / center_dist
            to_other_norm_y = to_other_y / center_dist

            # Check if other vehicle is IN FRONT of ego (not just facing same way)
            forward_dot = ego_forward.x * to_other_norm_x + ego_forward.y * to_other_norm_y

            if forward_dot < 0.8:  # ~45° cone
                return None

            # is it in our lane?
            ego_right_x = -ego_forward.y
            ego_right_y = ego_forward.x
            lateral_offset = abs(ego_right_x * to_other_x + ego_right_y * to_other_y)

            if lateral_offset > 2.5:  # More than ~1 lane width away
                return None



            # Bumper-to-bumper distance
            try:
                ego_extent = self.__ego.bounding_box.extent.x
                other_extent = vehicle.bounding_box.extent.x
                bumper_dist = max(0.0, center_dist - ego_extent - other_extent)
            except Exception:
                bumper_dist = max(0.0, center_dist - 6)

            return (bumper_dist, vehicle)

        except RuntimeError:
            return None

    def get_state(self) -> VehicleState:
        # Early exit if ego is invalid

        #if self._cached_traffic_lights is not None:
         #   for l in self._cached_traffic_lights:
          #      self._light_up_actor_box(l)

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
                data = self._safe_get_vehicle_data(v, ego_loc, ego_forward)
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
        for dist, vehicle in sorted(vehicle_data, key=lambda x: x[0]):
            if dist < smallest_dist:
                smallest_dist = dist
                break

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
            target_light_actor = self._get_trafficlight_simple(ego_loc, ego_forward)

            if target_light_actor:
                try:
                    self._light_up_actor_box(target_light_actor)

                    light_loc : carla.Location = target_light_actor.get_location()

                    to_light_x = light_loc.x - ego_loc.x
                    to_light_y = light_loc.y - ego_loc.y
                    longitudinal_dist = to_light_x * ego_forward.x + to_light_y * ego_forward.y

                    try:
                        ego_front_extent = self.__ego.bounding_box.extent.x
                    except Exception:
                        ego_front_extent = 2.5

                    traffic_light_dist_m = max(0.0, longitudinal_dist - ego_front_extent)
                    traffic_light_color = self._get_light_color_enum(target_light_actor.get_state())


                except RuntimeError:
                    pass
        except Exception:
            pass

        self.counter += 1

        # Update calculators
        self.__g_force_ego_calculator.update_speed(ego_velocity_ms)
        self.__relative_speed_lead_calculator.update_speed(result_dist_m)
        self.__speed_light_calculator.update_speed(traffic_light_dist_m)

        ego_g_force = self.__g_force_ego_calculator.get_latest_g_force() or 0
        relative_speed_ms = self.__relative_speed_lead_calculator.get_latest_g_force() or 0
        speed_light_ms = self.__speed_light_calculator.get_latest_g_force() or 0

        if self.counter % 300 == 0 or last_impact > 0.0:
            if keyboard.is_pressed('space'):
                time.sleep(0.2)
            logging.info(
                f"VS_LOG: "
                f"speed_ms={ego_velocity_ms:.2f}, "
                f"limit_ms={self.speed_limit / 3.6:.2f}, "
                f"lead_dist_m={result_dist_m:.2f}, "
                f"safe_dist_m={safe_distance_m:.2f}, "
                f"crash={last_impact}, "
                f"light_color={traffic_light_color}, "
                f"light_dist_m={traffic_light_dist_m:.2f}, "
                f"g_force={ego_g_force:.2f}, "
                f"rel_speed_ms={relative_speed_ms:.2f}, "
                f"light_speed_ms={speed_light_ms:.2f}"
            )

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