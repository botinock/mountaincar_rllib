from math import sqrt
import gymnasium as gym

from typing import Tuple, TypeVar

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")

class RewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, kind: str = 'none'):
        super().__init__(env)
        match kind:
            case 'none' | 'v' | 'y':
                self.kind = kind
            case _:
                raise ValueError("kind must be in none or distance")

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        obs, info = super().reset(**kwargs)
        match self.kind:
            case 'y':
                self.prev_obs = obs
        return obs, info


    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        match self.kind:
            case 'v':
                v = obs[1]
                reward += abs(v)
            case 'y':
                dx = abs(obs[0] - self.prev_obs[0]) 
                v = obs[1]
                dy = sqrt(abs(v**2 - dx**2))
                reward += abs(dy)
                self.prev_obs = obs
        return obs, reward, terminated, truncated, info