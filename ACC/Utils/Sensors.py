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


    def cleanup(self):

        if self.__collision_sensor is not None:
            self.__collision_sensor.destroy()

        self.__collision_sensor = None
        self.__ego = None
        self.__world = None
        self.__map = None

    def _get_light_color_enum(self, carla_state):
        """Maps CARLA TrafficLightState to your LightColors Enum"""
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

            # 2. Iterate ALL traffic light actors to find the one closest to this landmark
            #    (Since ID matching failed, we match by distance)
            all_traffic_lights = self.__world.get_actors().filter('traffic.traffic_light')

            closest_dist = float('inf')

            for tl_actor in all_traffic_lights:
                # Calculate distance between the Map Landmark and the Simulation Actor
                dist = tl_actor.get_location().distance(landmark_loc)

                # If it's within a small margin (e.g., 2 meters), it's the one!
                if dist < 2.0:
                    target_light_actor = tl_actor
                    break

                # Track closest just for debugging
                if dist < closest_dist:
                    closest_dist = dist

        return target_light_actor


    def get_state(self) -> VehicleState:

        ego_transform = self.__ego.get_transform()
        ego_loc = ego_transform.location

        vehicles = self.__world.get_actors().filter('vehicle.*')

        dist_calc = lambda l: math.sqrt((l.x - ego_loc.x)**2 + (l.y - ego_loc.y)**2 + (l.z - ego_loc.z)**2)

        vehicles = [(dist_calc(x.get_location()), x) for x in vehicles if x.id != self.__ego.id]


        ego_velocity_vec: Vector3D = self.__ego.get_velocity()
        ego_velocity_ms = ego_velocity_vec.length()

        safe_distance_m = self.__safe_time_distance_seconds * ego_velocity_ms
        has_crashed = self.__collision_sensor.get_last_impact() > 0.0

        smallest_dist = self.min_dist
        dists = []
        for dist, vehicle in sorted(vehicles):
            dot = vehicle.get_transform().get_forward_vector().dot(ego_transform.get_forward_vector()) # to see if the car is pointing the same way as the ego
            if smallest_dist > dist and dot > 0.8:
                smallest_dist = dist
            dists.append(smallest_dist)

        if len(vehicles) == 0:
            dists.append(smallest_dist)

        result_dist_m = min(dists)


        if self.override_speed_limit and self.counter == 2500:
            self.counter = 0
            self.speed_limit = random.randint(10, 140)
        elif not self.override_speed_limit:
            self.speed_limit = self.__ego.get_speed_limit()

        if self.speed_limit == 0.0:
            self.speed_limit = 30

        # traffic lights dist
        ego_waypoint = self.__map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

        target_light_actor = self._get_trafficlight(ego_waypoint)
        traffic_light_dist_m = self.min_dist
        traffic_light_color = LightColors.green

        # Use the actor if found
        if target_light_actor:
            traffic_light_dist_m = dist_calc(target_light_actor.get_location())
            traffic_light_color = self._get_light_color_enum(target_light_actor.get_state())


        if self.counter % 300 == 0:
            logging.info(f"speed: {ego_velocity_ms * 3.6}km/h, speed lim: {self.speed_limit} km/h, distance to nearest: {smallest_dist}m, safe dist: {safe_distance_m}m, CRASH: {has_crashed}")

        self.counter += 1

        self.__g_force_ego_calculator.update_speed(ego_velocity_ms)
        self.__relative_speed_lead_calculator.update_speed(result_dist_m)
        self.__speed_light_calculator.update_speed(traffic_light_dist_m)

        ego_g_force = self.__g_force_ego_calculator.get_latest_g_force()
        relative_speed_ms = self.__relative_speed_lead_calculator.get_latest_g_force()
        speed_light_ms = self.__speed_light_calculator.get_latest_g_force()

        if ego_g_force is None:
            ego_g_force = 0

        if relative_speed_ms is None:
            relative_speed_ms = 0

        if speed_light_ms is None:
            speed_light_ms = 0



        return VehicleState(speed_ms=ego_velocity_ms, speed_limit_ms=self.speed_limit / 3.6, lead_distance_m=result_dist_m,
                            safe_following_distance_m=safe_distance_m, hasCrashed=has_crashed,
                            light_color=traffic_light_color, light_dist_m=traffic_light_dist_m, g_force_ego=ego_g_force,
                            relative_speed_ms=relative_speed_ms, light_speed_ms=speed_light_ms)



class CarlaLeadStateSensor(StateSensor):

    def __init__(self, ego_vehicle: carla.Actor, lead: carla.Actor = None):
        self.__ego = ego_vehicle
        self.__lead = lead

        self.__safe_time_distance_seconds = 2
        self.counter = 0
        self.__collision_sensor = CollisionSensor(ego_vehicle)
        self.min_dist = 500


    def _get_light_color_enum(self, carla_state):
        """Maps CARLA TrafficLightState to your LightColors Enum"""
        if carla_state == carla.TrafficLightState.Red:
            return LightColors.red
        elif carla_state == carla.TrafficLightState.Yellow:
            return LightColors.orange
        else:
            return LightColors.green


    def get_state(self) -> VehicleState:
        ego_transform = self.__ego.get_transform()
        lead_transform = self.__lead.get_transform()
        ego_loc = ego_transform.location
        dist_calc = lambda l: math.sqrt((l.x - ego_loc.x)**2 + (l.y - ego_loc.y)**2 + (l.z - ego_loc.z)**2)

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


        # traffic lights dist
        ego_waypoint = self.__map.get_waypoint(ego_loc)
        lights_list = self.__world.get_traffic_lights_from_waypoint(ego_waypoint, 150.0)

        traffic_light_dist = self.min_dist
        traffic_light_color = LightColors.green

        if len(lights_list) > 0:
            target_light = lights_list[0]

            light_loc = target_light.get_location()
            traffic_light_dist = dist_calc(light_loc)

            traffic_light_color = self._get_light_color_enum(target_light.get_state())



        speed_limit = self.__ego.get_speed_limit()

        return VehicleState(speed_ms=ego_velocity_ms * 3.6, speed_limit_ms=speed_limit, lead_distance_m=[distance], safe_following_distance_m=safe_distance, hasCrashed=has_crashed, light_color=traffic_light_color, light_dist_m=traffic_light_dist)


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

            if self.sensor.is_listening:
                time.sleep(3)
                self.sensor.stop()

            time.sleep(3)

            if self.sensor.is_alive:
                self.sensor.destroy()

            self.sensor = None
        self._parent = None

class PygameUI(UI):
    pass