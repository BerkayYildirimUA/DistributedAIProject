class Scenario:


    def __init__(self, ego_car_bp_name: str = "random", lead_car_bp_name: str = "", map_name: str = "", number_of_npc: int = 0, turn_off_real_world_graphics = False, delta_seconds = 0.01, reward_crash=True, reward_geforce=True, reward_speed_limit=True, reward_safe_distance=True, reward_light=True):
        self.ego_car_bp_name = ego_car_bp_name
        self.lead_car_bp_name = lead_car_bp_name
        self.map = map_name
        self.number_of_npc = number_of_npc
        self.turn_off_real_world_graphics = turn_off_real_world_graphics
        self.delta_seconds = delta_seconds

        self.rewards = dict()
        self.rewards["reward_crash"] = reward_crash
        self.rewards["reward_geforce"] = reward_geforce
        self.rewards["reward_speed_limit"] = reward_speed_limit
        self.rewards["reward_safe_distance"] = reward_safe_distance
        self.rewards["reward_light"] = reward_light


        self.name = None