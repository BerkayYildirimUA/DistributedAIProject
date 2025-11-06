from __future__ import annotations

from typing import SupportsFloat, Any, Dict
import carla
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.core import RenderFrame, ActType, ObsType
from torch.backends.quantized import engine

from ACC.Engine.engine import Engine
from ACC.Utils.abstractions import ActionsEnum
from ACC.Utils.abstractions import VehicleState
from ACC.Utils.implementations import CarlaWorldStateSensor
from gymnasium import spaces
import traceback


class CarlaEnv(gym.Env[VehicleState, Dict[ActionsEnum, float]]):

    def __init__(self, args, scene):
        super().__init__()
        self.engine = Engine(args, scene)
        self.engine.connect_to_worlds()
        if not self.engine.setup():
            raise RuntimeError("Engine setup failed. Exiting.")
        self.sensor_real = CarlaWorldStateSensor(self.engine.ego.real, self.engine.duo_world.get_real_world())

        self.max_vehicles = 5
        self.observation_space = spaces.Box(
            low=np.array([0.0] * self.max_vehicles, dtype=np.float32),
            high=np.array([130.0] * self.max_vehicles, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

    def _array_to_action(self, action: np.ndarray) -> dict[ActionsEnum, float]:
        return {
            ActionsEnum.throttle: float(action[0]),
            ActionsEnum.brake: float(action[1]),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> \
            tuple[VehicleState, dict[str, Any]]:
        super().reset(seed=seed)
        self.engine.setup()

        self.sensor_real = CarlaWorldStateSensor(self.engine.ego.real, self.engine.duo_world.get_real_world())

        obs = self.sensor_real.get_state()
        info: dict[str, Any] = {}
        return obs, info

    def close(self):
        super().close()
        self.engine.cleanup()


    def _reward(self) -> SupportsFloat:

        #self.engine.ego.real

        self.sensor_real.get_state()




        return 0

    def step(self, action: dict[ActionsEnum, float]) -> tuple[VehicleState, SupportsFloat, bool, bool, dict[str, dict[ActionsEnum, float]]]:
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        try:
            mirror_frame, _ = self.engine.duo_world.tick()

            # apply control
            tm_control = self.engine.ego.get_mirror_control()

            action = self._array_to_action(action)
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

        obs = self.sensor_real.get_state()
        return obs, reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass




