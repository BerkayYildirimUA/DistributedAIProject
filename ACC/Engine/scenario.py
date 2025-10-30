import carla
from flatbuffers.flexbuffers import String


class Scenario:


    def __init__(self, ego_car_bp_name: String = "random", lead_car_bp_name: String = "", map_name: String = "", number_of_npc: int = 0, turn_off_real_world_graphics = False, delta_seconds = 0.01 ):
        self.ego_car_bp_name = ego_car_bp_name
        self.lead_car_bp_name = lead_car_bp_name
        self.map = map_name
        self.number_of_npc = number_of_npc
        self.turn_off_real_world_graphics = turn_off_real_world_graphics
        self.delta_seconds = delta_seconds