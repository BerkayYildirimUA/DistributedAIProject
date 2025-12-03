import carla

from ACC.Utils.abstractions import DecisionAgent, StateSensor


class SimpleAccAgent(DecisionAgent):

    def __init__(self, ego_vehicle: carla.Actor, sensor: StateSensor):
        self.__ego = ego_vehicle
        self.__sensor = sensor

    def make_decision(self, tm_control) -> carla.VehicleControl:

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
