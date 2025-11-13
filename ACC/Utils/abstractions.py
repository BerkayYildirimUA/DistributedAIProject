from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import carla
from enum import Enum


class LightColors(Enum):
    green = 0
    orange = 1
    red = 2

@dataclass
class VehicleState:
    speed: float
    speed_limit: float
    distances: List[float] #car in front of the Ego
    safe_following_distance: float
    hasCrashed: bool
    light_color: LightColors

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

class DecisionAgent(ABC):
    """Classes to make driving decisions. steering will be done by auto pilote, but acc will later be done with RL"""

    @abstractmethod
    def make_decision(self, temp) -> carla.VehicleControl:
        """get the next actions"""
    pass

class UI(ABC):
    """idk if this is needed. But UI stuff to display data while running"""
    pass