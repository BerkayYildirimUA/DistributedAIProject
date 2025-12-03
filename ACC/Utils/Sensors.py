import random
import time
import weakref
import collections

import carla
from carla import Vector3D
import math
import logging

from ACC.Utils.abstractions import StateSensor, UI, VehicleState, LightColors

class CarlaWorldStateSensor(StateSensor):

    def __init__(self, ego_vehicle: carla.Vehicle, world: carla.World):
        self.__ego = ego_vehicle
        self.__world = world

        self.__safe_time_distance_seconds = 3
        self.counter = 0
        self.__collision_sensor = CollisionSensor(ego_vehicle)

        self.override_speed_limit = False
        self.speed_limit = 0
    def cleanup(self):

        if self.__collision_sensor is not None:
            self.__collision_sensor.destroy()

        self.__collision_sensor = None

#    def reset(self, ego, world):
#
#        self.__ego = ego
#        self.__world = world
#        self.counter = 0
#        if self.__collision_sensor:
#            self.__collision_sensor.destroy()



    def get_state(self) -> VehicleState:

        ego_transform = self.__ego.get_transform()

        vehicles = self.__world.get_actors().filter('vehicle.*')

        dist = lambda l : math.sqrt((l.x - ego_transform.location.x)**2 + (l.y - ego_transform.location.y)
                             ** 2 + (l.z - ego_transform.location.z)**2)

        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != self.__ego.id]


        ego_velocity_vec: Vector3D = self.__ego.get_velocity()
        ego_velocity_ms = ego_velocity_vec.length()

        safe_distance = self.__safe_time_distance_seconds * ego_velocity_ms
        has_crashed = self.__collision_sensor.get_last_impact() > 0.0

        smallest_dist = 250
        dists = []
        for dist, vehicle in sorted(vehicles):
            dot = vehicle.get_transform().get_forward_vector().dot(ego_transform.get_forward_vector()) # to see if the car is pointing the same way as the ego
            if smallest_dist > dist and dot > 0.8:
                smallest_dist = dist
            dists.append(dist)

        if len(vehicles) == 0:
            dists.append(250)


        if self.override_speed_limit and self.counter == 2500:
            self.counter = 0
            self.speed_limit = random.randint(10, 140)
        elif not self.override_speed_limit:
            self.speed_limit = self.__ego.get_speed_limit()

        if self.speed_limit == 0.0:
            self.speed_limit = 30


        if self.counter % 300 == 0:
            logging.info(f"speed: {ego_velocity_ms * 3.6}km/h, speed lim: {self.speed_limit} km/h, distance to nearest: {smallest_dist}m, safe dist: {safe_distance}m, CRASH: {has_crashed}")

        self.counter += 1


        return VehicleState(speed=ego_velocity_ms * 3.6, speed_limit=self.speed_limit, distances=dists, safe_following_distance=safe_distance, hasCrashed=has_crashed, light_color=LightColors.green)



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

            if self.sensor.is_listening:
                time.sleep(3)
                self.sensor.stop()

            time.sleep(3)

            if self.sensor.is_alive:
                self.sensor.destroy()

            self.sensor = None

class PygameUI(UI):
    pass