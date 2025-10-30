import carla
from carla import Vector3D
import math

from ACC.Utils.abstractions import StateSensor, DecisionAgent, UI, VehicleState

class CarlaWorldStateSensor(StateSensor):

    def __init__(self, ego_vehicle: carla.Actor, world: carla.World):
        self.__ego = ego_vehicle
        self.__world = world

        self.__safe_time_distance_seconds = 2
        self.counter = 0

    def get_state(self) -> VehicleState:
        ego_transform = self.__ego.get_transform()

        vehicles = self.__world.get_actors().filter('vehicle.*')

        dist = lambda l : math.sqrt((l.x - ego_transform.location.x)**2 + (l.y - ego_transform.location.y)
                             ** 2 + (l.z - ego_transform.location.z)**2)

        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != self.__ego.id]


        ego_velocity_vec: Vector3D = self.__ego.get_velocity()
        ego_velocity_ms = ego_velocity_vec.length()

        safe_distance = self.__safe_time_distance_seconds * ego_velocity_ms

        smallest_dist = 400
        dists = []
        for dist, vehicle in sorted(vehicles):
            if smallest_dist > dist:
                smallest_dist = dist
            dists.append(dist)

        if self.counter == 100:
            self.counter = 0
            print(f"speed: {ego_velocity_ms * 3.6} km/h, distance to nearest: {smallest_dist}m, safe dist: {safe_distance}m")
        else:
            self.counter += 1

        return VehicleState(speed=ego_velocity_ms * 3.6, speed_limit=360, distances=dists, safe_following_distance=safe_distance)

class CarlaLeadStateSensor(StateSensor):

    def __init__(self, ego_vehicle: carla.Actor, lead: carla.Actor = None):
        self.__ego = ego_vehicle
        self.__lead = lead

        self.__safe_time_distance_seconds = 2
        self.counter = 0

    def get_state(self) -> VehicleState:
        ego_transform = self.__ego.get_transform()
        lead_transform = self.__lead.get_transform()

        distance = ego_transform.location.distance(lead_transform.location)

        ego_velocity_vec: Vector3D = self.__ego.get_velocity()
        ego_velocity_ms = ego_velocity_vec.length()

        safe_distance = self.__safe_time_distance_seconds * ego_velocity_ms

        if self.counter == 100:
            self.counter = 0
            print(f"speed: {ego_velocity_ms * 3.6} km/h, distance: {distance}m, safe dist: {safe_distance}m")
        else:
            self.counter += 1

        return VehicleState(speed=ego_velocity_ms * 3.6, speed_limit=360, distances=[distance], safe_following_distance=safe_distance)

class SimpleAccAgent(DecisionAgent):

    def __init__(self, ego_vehicle: carla.Actor, sensor: StateSensor):
        self.__ego = ego_vehicle
        self.__sensor = sensor

    def make_decision(self, temp) -> carla.VehicleControl:

        tm_control = temp

        data = self.__sensor.get_state()

        temp_break = 0.0
        temp_throttle = 0.0
        hand_break = False

        min_dist = min(data.distances)

        if data.speed < data.speed_limit and min_dist > data.safe_following_distance:
            temp_throttle = 0.6
            temp_break = 0
        else:
            temp_throttle = 0
            temp_break = 1

        if min_dist < 10:
            hand_break = True
            temp_throttle = 0
            temp_break = 1




        final_control = carla.VehicleControl(
            throttle=temp_throttle,
            brake=temp_break,
            steer=tm_control.steer,
            hand_brake=hand_break,
            reverse=tm_control.reverse,
            manual_gear_shift=tm_control.manual_gear_shift,
            gear=tm_control.gear
        )

        return final_control


class PygameUI(UI):
    pass