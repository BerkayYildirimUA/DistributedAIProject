from __future__ import annotations

import logging
import math
import random
from typing import SupportsFloat, Any, Dict
import carla
import gymnasium as gym
import numpy as np
import torch
from gymnasium.core import RenderFrame

from ACC.Engine.engine import Engine
from ACC.Utils.GForce_Class import GForceCalculator
from ACC.Utils.abstractions import ActionsEnum
from ACC.Utils.abstractions import VehicleState
from ACC.Utils.Sensors import CarlaWorldStateSensor
from ACC.Engine.scenario import Scenario
from gymnasium import spaces
import traceback

from mushroom_rl.utils.spaces import Box
from mushroom_rl.core import MDPInfo

from typing import Optional

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

        self.max_vehicles = 2

        self.eng_args = args
        self.eng_scene: Optional[Scenario] = scene

        # Complete observation space: speed + speed_limit + distances (5) + safe_distance + crashed + light_color
        # Total: 11 values
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0,  # speed
                 0.0,  # speed_limit
                 0.0,  # speed_ratio
                 0.0, 0.0,  # distances (5 vehicles)
                 0.0,  # safe_following_distance
                 0.0,  # hasCrashed (0 or 1)
                 0.0,  # light_color (0, 1, 2)
                 -1.0],  #steering
                dtype=np.float32
            ),
            high=np.array(
                [1.5,  # speed
                 1.5,  # speed_limit
                 1.5,  # speed ratio
                 1.0, 1.0,  # distances (max 1000m)
                 1.0,  # safe_following_distance
                 1.0,  # hasCrashed
                 1.0, # light_color
                 1.0], #steering
                dtype=np.float32
            ),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32,
        )

        self.__g_force_calculator = GForceCalculator(self.engine.delta_seconds)

        # MDPInfo for Mushroom-RL compatibility
        self._mdp_info = MDPInfo(
            observation_space=self.observation_space,
            action_space=self.action_space,
            gamma=0.99
            ,horizon=int(self.eng_args.horizon)
        )

        self.lead_speed_limit = 0

    def _vehicle_state_to_array(self, state: VehicleState) -> np.ndarray:
        distances = state.distances if state.distances else []
        padded_distances = list(distances) + [1000.0] * (self.max_vehicles - len(distances))
        padded_distances = padded_distances[:self.max_vehicles]

        #normilze
        norm_speed = state.speed / 130
        speed_ratio = state.speed / (state.speed_limit + 1e-5)
        norm_limit = state.speed_limit / 130.0

        norm_distances = np.array(padded_distances, dtype=np.float32) / 1000.0
        norm_distances = np.clip(norm_distances, 0.0, 1.0)

        norm_safe_dist = state.safe_following_distance / 100.0

        norm_light = float(state.light_color.value) / 2.0

        norm_steering = float(state.steering_dir)

        obs = np.array([
            norm_speed,
            norm_limit,
            speed_ratio,
            *norm_distances,
            norm_safe_dist,
            1.0 if state.hasCrashed else 0.0,
            norm_light,
            norm_steering
        ], dtype=np.float32)

        return obs

    def _array_to_action(self, action: np.ndarray) -> dict[ActionsEnum, float]:
        val = float(action[0])

        throttle = 0.0
        brake = 0.0

        if action >= -0.5:
            throttle = (val + 0.5) / 1.5
        else:
            brake = abs(val + 0.5) / 0.5

        return {
            ActionsEnum.throttle: throttle,
            ActionsEnum.brake: brake,
        }

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


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> \
            tuple[np.ndarray, dict[str, Any]]:
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
        if self.sensor_real is not None:
            current_speed_limit = self.sensor_real.speed_limit
            self.sensor_real.cleanup()
            self.sensor_real = None

        self.sensor_real = CarlaWorldStateSensor(self.engine.ego.real, self.engine.duo_world.get_real_world())

        if self.eng_args.random_speed_limit:
            self.sensor_real.override_speed_limit = True
            if current_speed_limit != 0.0:
                self.sensor_real.speed_limit = current_speed_limit

        state = self.sensor_real.get_state()
        state.steering_dir = 0.0
        obs = self._vehicle_state_to_array(state)
        info: dict[str, Any] = {}
        self.__g_force_calculator = GForceCalculator(self.engine.delta_seconds)

        return obs, info

    def close(self):
        super().close()

        self.engine.cleanup()


    def _reward(self, state : VehicleState) -> SupportsFloat:

        #self.engine.ego.real
        self.__g_force_calculator.update_speed(state.speed)
        g_force = self.__g_force_calculator.get_latest_g_force()

        rewards_dict = self.eng_scene.rewards
        #print(rewards_dict)
        use_crash = rewards_dict.get("reward_crash", True)
        use_geforce = rewards_dict.get("reward_geforce", True)
        use_speed = rewards_dict.get("reward_speed_limit", True)
        use_dist = rewards_dict.get("reward_safe_distance", True)

        ############### CRASH ###############
        if use_crash and state.hasCrashed:
            logging.info("Car Crashed!")
            return -15

        ############### G-FORCE ###############
        #TODO: simply, make continuous
        r_geforce = 0
        if use_geforce:
            if g_force is not None: # https://www.sciencedirect.com/science/article/pii/S0003687022002046?via%3Dihub
                if abs(g_force) < (0.56 / 9.81):
                    r_geforce = 2
                elif  abs(g_force) < (1.23 / 9.81):
                    r_geforce = 1
                elif  abs(g_force) < (2.12 / 9.81):
                    r_geforce = 2
                else:
                    r_geforce = math.exp(9.81) * (abs(g_force) - 2)


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
        if use_dist:
            if state.distances is not None and len(state.distances) > 0:
                min_front_distance = min(state.distances)
                safe_distance = state.safe_following_distance

                safety_margin = min_front_distance / (safe_distance + 1e-5)

                r_dist = 1 - 2 * math.exp(-4 * safety_margin * safety_margin)



        #logging.info(f"rewards: {reward}")
        #logging.info(f"  [Crash]         : {'ON' if use_crash else 'OFF'}")
        #logging.info(f"  [G-Force]       : {'ON' if use_geforce else 'OFF'}")
        #logging.info(f"  [Speed Limit]   : {'ON' if use_speed else 'OFF'}")
        #logging.info(f"  [Safe Distance] : {'ON' if use_dist else 'OFF'}")

        reward = r_dist + r_geforce + r_speed

        return reward

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

            if not self.engine.lead.is_alive():
                logging.debug("Lead pair incomplete. Attempting revival...")
                if self.engine.revive_lead_pair(self.lead_speed_limit):
                    logging.debug("Ego successfully revived. Continuing step.")
                else:
                    raise RuntimeError("Ego vehicle disappeared and revival failed.")



            # apply control
            tm_control = self.engine.ego.get_mirror_control()
            steering_dir = tm_control.steer
            action = self._array_to_action(action)

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
            reward = self._reward(state)
            obs = self._vehicle_state_to_array(state)

            self.engine.duo_world.tick()


            #this creates an infinite road to drive on
            if self.engine.map_name == "CUSTOM_STRAIGHT":
                transform = self.engine.ego.real.get_transform()
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
        return obs, reward, done, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


    def stop(self):
        return self.env.close()

from collections import deque
class FrameStackWrapper:
    def __init__(self, env, num_stack=4):
        self.env = env
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        # Calculate new observation space size
        # We assume flat observations here
        low = np.tile(env.observation_space.low, num_stack)
        high = np.tile(env.observation_space.high, num_stack)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = env.action_space

    @property
    def info(self):
        return MDPInfo(
            observation_space=self.observation_space,
            action_space=self.action_space,
            gamma=self.env.info.gamma,
            horizon=self.env.info.horizon
        )

    def _get_obs(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames))

    def reset(self, state=None):
        obs = self.env.reset(state)

        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def close(self):
        return self.env.close()

    def stop(self):
        return self.env.close()