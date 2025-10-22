from ACC.PythonAPI.carla.agents.navigation.behavior_agent import BehaviorAgent
from ACC.Utils.abstractions import *

class ProjectAgent(BehaviorAgent):
    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None, decision_agent: DecisionAgent = None):
        super().__init__(vehicle, behavior, opt_dict, map_inst, grp_inst)
        if decision_agent is None:
            raise Exception("No DecisionAgent is None")

        self.__decision_agent = decision_agent




    def run_step(self, debug=False):
        nav_control = super().run_step(debug)

        return self.__decision_agent.make_decision(nav_control)