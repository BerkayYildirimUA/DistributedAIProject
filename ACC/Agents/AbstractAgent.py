
from abc import ABC, abstractmethod
import carla

class AbstractAgent(ABC):


    @abstractmethod
    def make_decision(self) -> carla.VehicleControl:
        """get state of the car"""
        pass