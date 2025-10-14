from turtle import Vec2D

import carla
import pygame
from carla import Vector3D

from abstractions import StateSensor, DecisionAgent, UI, VehicleState


class CarlaStateSensor(StateSensor):

    def __init__(self, ego_vehicle: carla.Actor, lead: carla.Actor):
        self.__ego = ego_vehicle
        self.__lead = lead
        self.__safe_time_distance_seconds = 2

    def get_state(self) -> VehicleState:
        ego_transform = self.__ego.get_transform()
        lead_transform = self.__lead.get_transform()

        distance = ego_transform.location.distance(lead_transform.location)

        ego_velocity_vec: Vector3D = self.__ego.get_velocity()
        ego_velocity = ego_velocity_vec.length()


        print(ego_velocity, distance)

        return VehicleState(speed=ego_velocity, speed_limit=80, distance_to_lead=distance, safe_following_distance=10)

class SimpleAccAgent(DecisionAgent):

    def __init__(self, ego_vehicle: carla.Actor, sensor: CarlaStateSensor):
        self.__ego = ego_vehicle
        self.__sensor = sensor

    def make_decision(self) -> carla.VehicleControl:

        tm_control = self.__ego.get_control()

        data = self.__sensor.get_state()

        temp_break = 0.0
        temp_throttle = 0.0

        if data.speed < data.speed_limit and data.distance_to_lead > data.safe_following_distance:
            temp_throttle = 100
            temp_break = 0
        else:
            hand_break = True
            #temp_throttle = 0
            #temp_break = 1

        hand_break = False
        if data.distance_to_lead < data.safe_following_distance:
            hand_break = True


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