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

        self.__safe_time_distance_seconds = 2
        self.counter = 0
        self.__collision_sensor = CollisionSensor(ego_vehicle)

    def cleanup(self):

        if self.__collision_sensor:
            self.__collision_sensor.cleanup()

    def reset(self, ego, world):

        self.__ego = ego
        self.__world = world
        self.counter = 0
        if self.__collision_sensor:
            self.__collision_sensor.reset()





    def get_state(self) -> VehicleState:

        ego_transform = self.__ego.get_transform()

        vehicles = self.__world.get_actors().filter('vehicle.*')

        dist = lambda l : math.sqrt((l.x - ego_transform.location.x)**2 + (l.y - ego_transform.location.y)
                             ** 2 + (l.z - ego_transform.location.z)**2)

        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != self.__ego.id]


        ego_velocity_vec: Vector3D = self.__ego.get_velocity()
        ego_velocity_ms = ego_velocity_vec.length()

        safe_distance = self.__safe_time_distance_seconds * ego_velocity_ms
        has_crashed = self.__collision_sensor.has_collided

        smallest_dist = 400
        dists = []
        for dist, vehicle in sorted(vehicles):
            dot = vehicle.get_transform().get_forward_vector().dot(ego_transform.get_forward_vector()) # to see if the car is pointing the same way as the ego
            if smallest_dist > dist and dot > 0.8:
                smallest_dist = dist
            dists.append(dist)

        if len(vehicles) == 0:
            dists.append(400)

        speed_limit = self.__ego.get_speed_limit()

        if speed_limit == 0.0:
            speed_limit = 30


        if self.counter == 100:
            self.counter = 0
            logging.info(f"speed: {ego_velocity_ms * 3.6}km/h, speed lim: {speed_limit} km/h, distance to nearest: {smallest_dist}m, safe dist: {safe_distance}m")
        else:
            self.counter += 1




        return VehicleState(speed=ego_velocity_ms * 3.6, speed_limit=speed_limit, distances=dists, safe_following_distance=safe_distance, hasCrashed=has_crashed, light_color=LightColors.green)



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

        has_crashed = self.__collision_sensor.has_collided

        speed_limit = self.__ego.get_speed_limit()

        if speed_limit == 0.0:
            speed_limit = 30

        if self.counter == 100:
            self.counter = 0
            logging.info(f"speed: {ego_velocity_ms * 3.6}km/h, speed lim: {speed_limit} km/h, distance to nearest: {distance}m, safe dist: {safe_distance}m")
        else:
            self.counter += 1

        speed_limit = self.__ego.get_speed_limit()

        return VehicleState(speed=ego_velocity_ms * 3.6, speed_limit=speed_limit, distances=[distance], safe_following_distance=safe_distance, hasCrashed=has_crashed, light_color=LightColors.green)


#code form carla examples, from "automatic_control.py"
class CollisionSensor(object):
    """ Class for collision sensors """

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self.has_collided = False
        self._parent = parent_actor

        world = self._parent.get_world()

        world.tick()

        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)


        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        self.has_collided = True
        if len(self.history) > 4000:
            self.history.pop(0)

    def cleanup(self):
        """Explicitly stop listening to prevent Stream errors"""
        if self.sensor is not None:
            if self.sensor.is_listening:
                self.sensor.stop()
            if self.sensor.is_alive:
                self.sensor.destroy()
            self.sensor = None

    def reset(self):
        """Clear the crash data for the next episode"""
        self.history = []
        self.has_collided = False

class PygameUI(UI):
    pass