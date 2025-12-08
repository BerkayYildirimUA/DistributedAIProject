
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import carla
from enum import Enum

import numpy

class LightColors(Enum):
    green = 0
    orange = 1
    red = 2

@dataclass
class VehicleState: #maybe add steering direction?
    speed_ms: float #speed of the geo
    speed_limit_ms: float #speed limit the geo should follow
    lead_distance_m: float #distance to the car in front
    safe_following_distance_m: float #safe driving distance from the car in front
    hasCrashed: bool #has crashed or not
    light_color: LightColors #color of lights
    light_dist_m: float #distance to nearest light
    light_speed_ms: float #how fast we are approaching that light
    g_force_ego: float #the g force the ego is experincesing
    relative_speed_ms: float #the speed diff between ego and car in front

class ActionsEnum(Enum):
    brake = 1
    throttle = 2

class StateSensor(ABC):
    """Class that returns the state of the world. Could be made to use sensors data in the future, but for now just calc via carla's build in methods"""

    @abstractmethod
    def get_state(self) -> VehicleState:
        """get state of the car"""
        pass


    pass

class AbstractDecisionAgent(ABC):
    """Classes to make driving decisions. steering will be done by auto pilote, but acc will later be done with RL"""

    @abstractmethod
    def make_decision(self, temp) -> carla.VehicleControl:
        """get the next actions"""
    pass

class UI(ABC):
    """idk if this is needed. But UI stuff to display data while running"""
    pass