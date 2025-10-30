import carla
from flatbuffers.flexbuffers import String


class Scenario:


    def __init__(self, ego_car: carla.ActorBlueprint, lead_car: carla.ActorBlueprint = None, map_name: String = "", number_of_npc: int = 0, turn_off_real_world_graphics = False, tick_rate = 0.01 ):



        self.ego_car = ego_car
        self.lead_car = lead_car
        self.map = map_name
        self.number_of_npc = number_of_npc
        self.turn_off_real_world_graphics = turn_off_real_world_graphics
