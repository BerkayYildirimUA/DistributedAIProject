from __future__ import annotations

from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.core import RenderFrame, ActType, ObsType


class CarlaEnv(gym.Env):

    def __init__(self):
        super().__init__()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def close(self):
        super().close()

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        pass

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass