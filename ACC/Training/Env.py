from __future__ import annotations

import logging
import math
import random
from typing import SupportsFloat, Any, Dict
import carla
import gymnasium as gym
import numpy as np
import wandb
from carla import Vector3D
from gymnasium.core import RenderFrame

from ACC.Engine.engine import Engine
from ACC.Utils.abstractions import ActionsEnum
from ACC.Utils.abstractions import VehicleState
from ACC.Utils.Sensors import CarlaWorldStateSensor
from ACC.Engine.scenario import Scenario
from gymnasium import spaces
import traceback

from mushroom_rl.core import MDPInfo

from typing import Optional


def vehicle_state_to_array(state: VehicleState) -> np.ndarray:


    #normilze
    norm_speed = state.speed / 130
    speed_ratio = state.speed / (state.speed_limit + 1e-5)
    norm_limit = state.speed_limit / 130.0

    norm_distance = np.clip(state.lead_distance / 500.0, 0.0, 1.0)

    norm_safe_dist = state.safe_following_distance / 150.0

    norm_light = float(state.light_color.value) / 2.0

    norm_light_dist = float(state.light_dist) / 500

    norm_light_speed =np.clip(state.light_speed / 10, -1.5, 1.5)
    norm_speed_lead = np.clip(state.speed_lead / 5, -1.5, 1.5)
    norm_acc_ego = np.clip(state.acc_ego / 5, -1.5, 1.5)

    obs = np.array([
        norm_speed,
        norm_limit,
        speed_ratio,
        norm_distance,
        norm_safe_dist,
        1.0 if state.hasCrashed else 0.0,
        norm_light,
        norm_light_dist,
        norm_light_speed,
        norm_speed_lead,
        norm_acc_ego
    ], dtype=np.float32)

    return obs


def array_to_action(action: np.ndarray) -> Dict[ActionsEnum, float]:
    val = float(action[0])

    throttle = 0.0
    brake = 0.0

    if val >= -0.5:
        throttle = (val + 0.5) / 1.5
    else:
        brake = abs(val + 0.5) / 0.5

    return {
        ActionsEnum.throttle: throttle,
        ActionsEnum.brake: brake,
    }


class CarlaEnv(gym.Env[VehicleState, Dict[ActionsEnum, float]]):

    def __init__(self, args, scene):
        super().__init__()
        self.eng_args = None
        self.eng_scene: Optional[Scenario] = None


        self.engine = Engine(args, scene)
        self.engine.connect_to_worlds()
        if not self.engine.setup():
            raise RuntimeError("Engine setup failed. Exiting.")


        self.sensor_real = CarlaWorldStateSensor(self.engine.ego.real, self.engine.duo_world.get_real_world())

        self.eng_args = args
        self.eng_scene: Optional[Scenario] = scene

        # Complete observation space
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0,  # speed
                 0.0,  # speed_limit
                 0.0,  # speed_ratio
                 0.0,  # distance to lead
                 0.0,  # safe_following_distance
                 0.0,  # hasCrashed (0 or 1)
                 0.0,  # light_color (0, 1, 2)
                 0.0,  #light distance
                 -1.5, # norm_light_speed
                 -1.5,  # norm_speed_lead
                 -1.5], # norm_acc_ego
                dtype=np.float32
            ),
            high=np.array(
                [1.5,  # speed
                 1.5,  # speed_limit
                 1.5,  # speed ratio
                 1.0,  # distances (max 250m)
                 1.0,  # safe_following_distance
                 1.0,  # hasCrashed
                 1.0, # light_color
                 1.0, # light_dist (max 250m)
                 1.5, # norm_light_speed
                 1.5, # norm_speed_lead
                 1.5], # norm_acc_ego
                dtype=np.float32
            ),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32,
        )

        #self.__g_force_calculator = GForceCalculator(self.engine.delta_seconds)

        # MDPInfo for Mushroom-RL compatibility
        self._mdp_info = MDPInfo(
            observation_space=self.observation_space,
            action_space=self.action_space,
            gamma=0.99
            ,horizon=int(self.eng_args.horizon)
        )

        self.lead_speed_limit = 0

    @property
    def info(self):
        return self._mdp_info

    def set_rewards(self, reward_crash=True, reward_geforce=True, reward_speed_limit=True, reward_safe_distance=True):

        if self.eng_scene is not None:
            self.eng_scene.rewards["reward_crash"] = reward_crash
            self.eng_scene.rewards["reward_geforce"] = reward_geforce
            self.eng_scene.rewards["reward_speed_limit"] = reward_speed_limit
            self.eng_scene.rewards["reward_safe_distance"] = reward_safe_distance
        else:
            print("SOMETHING WRONG") #TODO: delete this debug


    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> \
            tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)



        if self.engine is not None:
            self.eng_args = self.engine.args
            self.eng_scene: Scenario = self.engine.scenario
            self.engine.cleanup()
            self.engine = None


        #self.set_rewards()

        self.engine = Engine(self.eng_args,self.eng_scene)
        self.engine.connect_to_worlds()
        self.engine.duo_world.tick()


        if not self.engine.setup():
            raise RuntimeError("Engine setup failed. Exiting.")


        #self.sensor_real.reset(self.engine.ego.real, self.engine.duo_world.get_real_world())
        current_speed_limit = 0.0
        counter = 0.0
        if self.sensor_real is not None:
            current_speed_limit = self.sensor_real.speed_limit
            counter = self.sensor_real.counter
            self.sensor_real.cleanup()
            self.sensor_real = None


        self.sensor_real = CarlaWorldStateSensor(self.engine.ego.real, self.engine.duo_world.get_real_world())
        self.sensor_real.counter = counter

        if self.eng_args.random_speed_limit:
            self.sensor_real.override_speed_limit = True
            if current_speed_limit != 0.0:
                self.sensor_real.speed_limit = current_speed_limit

        state = self.sensor_real.get_state()
        state.steering_dir = 0.0
        obs = vehicle_state_to_array(state)
        info: Dict[str, Any] = {}
        #self.__g_force_calculator = GForceCalculator(self.engine.delta_seconds)


        #self.engine.lead.set_mirror_velocity(Vector3D(x=self.lead_speed_limit, y=0, z=0))

        return obs, info

    def close(self):
        super().close()
        self.engine.cleanup()


    def _reward(self, state : VehicleState) -> tuple[float, Dict]:

        #self.engine.ego.real
        #self.__g_force_calculator.update_speed(state.speed)
        #g_force = self.__g_force_calculator.get_latest_g_force()



        rewards_dict = self.eng_scene.rewards
        #print(rewards_dict)
        use_crash = rewards_dict.get("reward_crash", True)
        use_geforce = rewards_dict.get("reward_geforce", False)
        use_speed = rewards_dict.get("reward_speed_limit", True)
        use_dist = rewards_dict.get("reward_safe_distance", True)

        ############### CRASH ###############
        r_crash = 0
        if use_crash and state.hasCrashed:
            logging.info("Car Crashed!")
            r_crash = -100

        ############### G-FORCE ###############
        #TODO: VERY OLD; NOT USED ATM, NEED TO BE FULLY REWORKED
        r_geforce = 0
        g_force_ego = state.acc_ego
        if use_geforce:
            if g_force_ego is not None: # https://www.sciencedirect.com/science/article/pii/S0003687022002046?via%3Dihub
                acc = g_force_ego * 9.81
                r_geforce = 0.645889 - 0.184412 * math.exp(1.01363 * acc)
                r_geforce = max(r_geforce, -2)



        ############### SPEED ###############
        r_speed = 0
        if use_speed:
            speed_diff = state.speed - state.speed_limit

            if speed_diff > 0:
                r_speed = 1.5 * math.exp(-(speed_diff ** 2) / 2.5) - 0.5 - 0.01 * speed_diff # maybe 0.05?
            else:
                r_speed = math.exp(-(speed_diff ** 2) / 25) + 0.01 * speed_diff


        ############### SAFE DISTANCE ###############
        r_dist = 0
        r_dist_weight = 1.5

        min_front_distance = 0.0
        safe_distance = 0.0
        safety_margin = 0.0
        if use_dist:
            if state.lead_distance is not None and state.lead_distance > 0:
                min_front_distance = state.lead_distance
                safe_distance = state.safe_following_distance

                safety_margin = min(min_front_distance / (safe_distance + 1e-5), 1000)

                if safety_margin >= 1:
                    r_dist = math.exp(-1 * (safety_margin-1))
                else:
                    r_dist = - 5 * ((safety_margin - 1) ** 2) + 1

                r_dist = r_dist_weight * r_dist



        #logging.info(f"rewards: {reward}")
        #logging.info(f"  [Crash]         : {'ON' if use_crash else 'OFF'}")
        #logging.info(f"  [G-Force]       : {'ON' if use_geforce else 'OFF'}")
        #logging.info(f"  [Speed Limit]   : {'ON' if use_speed else 'OFF'}")
        #logging.info(f"  [Safe Distance] : {'ON' if use_dist else 'OFF'}")

        total_reward = r_crash + r_dist + r_geforce + r_speed if r_crash == 0 else r_crash

        components = {
            "Reward/Total": total_reward,
            "Reward/Crash": r_crash,
            "Reward/GForce": r_geforce,
            "Reward/Speed": r_speed,
            "Reward/Distance": r_dist,
            "State/distance/safe_distance": safe_distance,
            "State/distance/safety_margin": safety_margin,


            "State/VehicleState/speed": state.speed,
            "State/VehicleState/speed_limit": state.speed_limit,
            "State/VehicleState/lead_distance": state.lead_distance,
            "State/VehicleState/safe_following_distance": state.safe_following_distance,
            "State/VehicleState/hasCrashed": state.hasCrashed,
            "State/VehicleState/light_color": state.light_color.value,
            "State/VehicleState/light_dist": state.light_dist,
            "State/VehicleState/light_speed": state.light_speed,
            "State/VehicleState/acc_ego": state.acc_ego,
            "State/VehicleState/speed_lead": state.speed_lead
        }

        return total_reward, components

    def step(self, action: np.ndarray) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, dict[ActionsEnum, float]]]:
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0

        try:

            if not self.engine.ego.is_alive():
                logging.debug("Ego pair incomplete. Attempting revival...")
                if self.engine.revive_ego_pair():
                    logging.debug("Ego successfully revived. Continuing step.")
                else:
                    raise RuntimeError("Ego vehicle disappeared and revival failed.")

            if self.engine.lead is not None:
                if not self.engine.lead.is_alive():
                    logging.debug("Lead pair incomplete. Attempting revival...")
                    if self.engine.revive_lead_pair(self.lead_speed_limit):
                        logging.debug("Ego successfully revived. Continuing step.")
                    else:
                        raise RuntimeError("Ego vehicle disappeared and revival failed.")



            # apply control
            tm_control = self.engine.ego.get_mirror_control()
            steering_dir = tm_control.steer
            action = array_to_action(action)

            #logging.info(f"throttle: {action[ActionsEnum.throttle]}, brake: {action[ActionsEnum.brake]}")

            final_control = carla.VehicleControl(
                throttle=action[ActionsEnum.throttle],
                brake=action[ActionsEnum.brake],
                steer=steering_dir,
                hand_brake=False,
                reverse=tm_control.reverse,
                manual_gear_shift=tm_control.manual_gear_shift,
                gear=tm_control.gear
            )

            self.engine.ego.apply_real_control(final_control)

            # apply goal
            if self.engine.lead is not None:
                self.engine.tm_mirror.set_path(self.engine.ego.mirror, [self.engine.lead.mirror.get_location()])

            # synchronization real npc with mirror npcs
            self.engine.synchronization_real_npc_with_mirror_npcs()

            # synchronization mirror ego with real ego
            self.engine.synchronization_mirror_ego_with_real_ego()

            # spectator
            self.engine.update_spectator()

            state = self.sensor_real.get_state()

            if state.hasCrashed:
                logging.info("Car Crashed!")
                terminated = True
            state.steering_dir = steering_dir

            reward, reward_components = self._reward(state)

            if self.engine.lead is not None:
                v = self.engine.lead.mirror.get_velocity() * 3.6
                lead_speed = v.length()
                reward_components["State/LeadSpeed"] = lead_speed
            else:
                reward_components["State/LeadSpeed"] = 0.0

            info.update(reward_components)


            obs = vehicle_state_to_array(state)

            self.engine.duo_world.tick()


            #this creates an infinite road to drive on
            if "CUSTOM_STRAIGHT" in self.engine.map_name:
                transform = self.engine.ego.real.get_transform()

                if self.engine.lead is not None:
                    lead_transform = self.engine.lead.mirror.get_transform()
                    dist_to_lead = lead_transform.location.x - transform.location.x

                    if dist_to_lead > 300:
                        target_catchup_speed = max(0.0, state.speed * random.uniform(0.5, 0.8))

                        # Only update if meaningful change to avoid spamming TM
                        if abs(self.lead_speed_limit - target_catchup_speed) > 1.0:
                            self.lead_speed_limit = target_catchup_speed
                            self.engine.tm_mirror.set_desired_speed(self.engine.lead.mirror, self.lead_speed_limit)

                if transform.location.x > 1000:
                    new_location = carla.Location(x=10, y=transform.location.y, z=transform.location.z)
                    self.engine.ego.real.set_location(new_location)
                    self.engine.ego.mirror.set_location(new_location)
                    if self.engine.lead is not None:
                        lead_transform = self.engine.lead.real.get_transform()
                        distances_to_ego = lead_transform.location.x - transform.location.x
                        lead_location = carla.Location(x=distances_to_ego + new_location.x, y=lead_transform.location.y, z=lead_transform.location.z)
                        self.engine.lead.real.set_location(lead_location)
                        self.engine.lead.mirror.set_location(lead_location)

                        self.lead_speed_limit = state.speed_limit + random.randint(-15, 15)

                        self.engine.tm_mirror.set_desired_speed(self.engine.lead.mirror, self.lead_speed_limit)





        except Exception as e:
            print(f"\nAn critical error occurred during step in traing env: {e}")
            traceback.print_exc()
            terminated = True
        return obs, reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass


class GymnasiumToGymWrapper:

    def __init__(self, env: CarlaEnv):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.step_count = 0

    @property
    def info(self):
        return self.env._mdp_info

    def reset(self, state=None):
        max_retries = 10

        for attempt in range(max_retries):
            try:
                obs, _ = self.env.reset()

                if self.env.engine.ego is not None and self.env.engine.ego.is_alive():
                    return obs

                logging.warning(f"Reset attempt {attempt}: Ego is dead or None. Retrying...")

            except Exception as e:
                logging.warning(f"Reset attempt {attempt} crashed: {e}. Retrying...")
                traceback.print_exc()

        raise RuntimeError("Critical: Failed to reset environment after 10 attempts.")

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if wandb.run is not None:
            log_dict = {k: v for k, v in info.items() if isinstance(v, (int, float))}
            log_dict['global_step'] = self.step_count
            wandb.log(log_dict)

        self.step_count += 1

        return obs, reward, done, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


    def stop(self):
        return self.env.close()

