import carla
from carla import Vector3D

from ACC.Utils.abstractions import StateSensor, DecisionAgent, UI, VehicleState


class CarlaStateSensor(StateSensor):

    def __init__(self, ego_vehicle: carla.Actor, lead: carla.Actor):
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

        return VehicleState(speed=ego_velocity_ms * 3.6, speed_limit=360, distance_to_lead=distance, safe_following_distance=safe_distance)

class SimpleAccAgent(DecisionAgent):

    def __init__(self, ego_vehicle: carla.Actor, sensor: CarlaStateSensor):
        self.__ego = ego_vehicle
        self.__sensor = sensor

    def make_decision(self, temp) -> carla.VehicleControl:

        tm_control = temp

        data = self.__sensor.get_state()

        temp_break = 0.0
        temp_throttle = 0.0
        hand_break = False

        if data.speed < data.speed_limit and data.distance_to_lead > data.safe_following_distance:
            temp_throttle = 0.6
            temp_break = 0
        else:
            temp_throttle = 0
            temp_break = 1

        if data.distance_to_lead < 10:
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