import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from config.configs import Config


class SimpleMLPFeatureExtractor(BaseFeaturesExtractor):
    """
    Box observation space용 단순 MLP: 두 개의 512 히든 레이어.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = Config().FEATURES_DIM):
        super().__init__(observation_space, features_dim)

        # Box observation space의 크기
        input_dim = observation_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

