from __future__ import annotations

import logging
import math
from typing import SupportsFloat, Any, Dict
import carla
import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from torch.backends.quantized import engine

from ACC.Engine.engine import Engine
from ACC.Utils.GForce_Class import GForceCalculator
from ACC.Utils.abstractions import ActionsEnum
from ACC.Utils.abstractions import VehicleState
from ACC.Utils.Sensors import CarlaWorldStateSensor
from gymnasium import spaces
import traceback

from mushroom_rl.utils.spaces import Box
from mushroom_rl.core import MDPInfo


class CarlaEnv(gym.Env[VehicleState, Dict[ActionsEnum, float]]):

    def __init__(self, args, scene):
        super().__init__()
        self.eng_args = None
        self.eng_scene = None

        self.engine = Engine(args, scene)
        self.engine.connect_to_worlds()
        if not self.engine.setup():
            raise RuntimeError("Engine setup failed. Exiting.")
        self.sensor_real = CarlaWorldStateSensor(self.engine.ego.real, self.engine.duo_world.get_real_world())

        self.max_vehicles = 5

        # Complete observation space: speed + speed_limit + distances (5) + safe_distance + crashed + light_color
        # Total: 10 values
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0,  # speed
                 0.0,  # speed_limit
                 0.0, 0.0, 0.0, 0.0, 0.0,  # distances (5 vehicles)
                 0.0,  # safe_following_distance
                 0.0,  # hasCrashed (0 or 1)
                 0.0],  # light_color (0, 1, 2)
                dtype=np.float32
            ),
            high=np.array(
                [130.0,  # speed
                 130.0,  # speed_limit
                 1000.0, 1000.0, 1000.0, 1000.0, 1000.0,  # distances (max 1000m)
                 100.0,  # safe_following_distance
                 1.0,  # hasCrashed
                 2.0],  # light_color
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
            gamma=0.99,
            horizon=1000
        )

    def _vehicle_state_to_array(self, state: VehicleState) -> np.ndarray:
        distances = state.distances if state.distances else []
        padded_distances = list(distances) + [1000.0] * (self.max_vehicles - len(distances))
        padded_distances = padded_distances[:self.max_vehicles]

        obs = np.array([
            state.speed,
            state.speed_limit,
            *padded_distances,
            state.safe_following_distance,
            1.0 if state.hasCrashed else 0.0,
            float(state.light_color.value)
        ], dtype=np.float32)

        return obs

    def _array_to_action(self, action: np.ndarray) -> dict[ActionsEnum, float]:
        action = float(action[0])

        throttle = 0.0
        brake = 0.0

        if action >= 0:
            throttle = abs(action)
        else:
            brake = abs(action)

        return {
            ActionsEnum.throttle: throttle,
            ActionsEnum.brake: brake,
        }

    @property
    def info(self):
        return self._mdp_info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> \
            tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)


        if engine is not None:
            self.eng_args = self.engine.args
            self.eng_scene = self.engine.scenario
            self.engine.cleanup()
            self.engine = None

        self.engine = Engine(self.eng_args,self.eng_scene)
        self.engine.connect_to_worlds()
        if not self.engine.setup():
            raise RuntimeError("Engine setup failed. Exiting.")

        self.sensor_real = CarlaWorldStateSensor(self.engine.ego.real, self.engine.duo_world.get_real_world())

        state = self.sensor_real.get_state()
        obs = self._vehicle_state_to_array(state)
        info: dict[str, Any] = {}
        self.__g_force_calculator = GForceCalculator(self.engine.delta_seconds)

        return obs, info

    def close(self):
        super().close()
        self.engine.cleanup()


    def _reward(self) -> SupportsFloat:

        #self.engine.ego.real

        state : VehicleState = self.sensor_real.get_state()

        self.__g_force_calculator.update_speed(state.speed)


        #crash
        reward = 0.0
        if state.hasCrashed:
            reward = -100
        else:
            reward += 1


        #geforce

        g_force = self.__g_force_calculator.get_latest_g_force()
        if g_force is not None: # https://www.sciencedirect.com/science/article/pii/S0003687022002046?via%3Dihub
            if abs(g_force) < (0.56 / 9.81):
                reward += 2
            elif  abs(g_force) < (1.23 / 9.81):
                reward += 1
            elif  abs(g_force) < (2.12 / 9.81):
                reward -= 2
            else:
                reward -= math.exp(9.81) * (abs(g_force) - 2)


        min_front_distance = min(state.distances) if state.distances and len(state.distances) > 0 else 1000.0

        #speed limit
        if state.speed > (state.speed_limit + 3):
            reward -= (state.speed - state.speed_limit - 1)
        elif state.speed < (state.speed_limit - 5) and min_front_distance > (state.safe_following_distance * 2): # Going too slow without reason
            speed_deficit = (state.speed_limit - 5) - state.speed
            penalty = speed_deficit / state.speed_limit  # Normalized penalty? Idk if this is the right way, just feels correct TODO double check
            reward -= 3 * penalty
        else:
            reward += 1
            if abs(g_force) < (1.23 / 9.81):
                reward += 10

        if state.distances is not None and len(state.distances) > 0:
            min_front_distance = min(state.distances)
            safe_distance = state.safe_following_distance

            if min_front_distance < safe_distance:
                penalty = math.exp((safe_distance - min_front_distance) / safe_distance) - 1
                reward -= 5 * penalty

        #logging.info(f"rewards: {reward}")
        return reward

    def step(self, action: np.ndarray) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, dict[ActionsEnum, float]]]:
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        try:
            mirror_frame, _ = self.engine.duo_world.tick()

            # apply control
            tm_control = self.engine.ego.get_mirror_control()

            action = self._array_to_action(action)

            #logging.info(f"throttle: {action[ActionsEnum.throttle]}, brake: {action[ActionsEnum.brake]}")

            final_control = carla.VehicleControl(
                throttle=action[ActionsEnum.throttle],
                brake=action[ActionsEnum.brake],
                steer=tm_control.steer,
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
        except Exception as e:
            print(f"\nAn critical error occurred during step in traing env: {e}")
            traceback.print_exc()
            terminated = True

        reward = self._reward()

        state = self.sensor_real.get_state()
        obs = self._vehicle_state_to_array(state)

        return obs, reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass


class GymnasiumToGymWrapper:

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    @property
    def info(self):
        return self.env._mdp_info

    def reset(self, state=None):
        obs, _ = self.env.reset()
        return obs

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
