import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from config.configs import Config

class ResidualBlock(nn.Module):
    """
    Residual Block: x = x + Linear(ReLU(Linear(x)))
    """
    def __init__(self, size: int):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(size, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return identity + out

class ResidualFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor that processes the concatenated HER dictionary
    using a Residual MLP architecture.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = Config().FEATURES_DIM):
        # We call super with the observation space and the desired features dim
        super().__init__(observation_space, features_dim)
        
        # Calculate the total dimension of the flattened dictionary
        total_concat_size = 0
        for key in sorted(observation_space.spaces.keys()):
            subspace = observation_space.spaces[key]
            if isinstance(subspace, gym.spaces.Box):
                total_concat_size += int(torch.prod(torch.tensor(subspace.shape)))
            else:
                raise NotImplementedError("Only Box spaces are supported for this extractor")

        self.initial_projection = nn.Linear(total_concat_size, Config().HIDDEN_SIZE)
        self.initial_activation = nn.ReLU()
        
        # Residual Blocks
        self.res_block1 = ResidualBlock(Config().HIDDEN_SIZE)
        self.res_block2 = ResidualBlock(Config().HIDDEN_SIZE)
        
        self.final_projection = nn.Linear(Config().HIDDEN_SIZE, features_dim)
        self.final_activation = nn.ReLU()

    def forward(self, observations: dict) -> torch.Tensor:
        # Concatenate all dictionary entries (observation, achieved_goal, desired_goal)
        # Note: SB3 passes observations as a dictionary of tensors.
        # We need to ensure a consistent order. 
        # Common HER keys: 'observation', 'achieved_goal', 'desired_goal'.
        # We'll sort keys to ensure consistency with __init__ calculation if we iterated there.
        # Actually in __init__, observation_space.spaces.items() iteration order is usually consistent but
        # strictly speaking we should sort to be safe or rely on SB3's guarantee.
        # Let's concatenate values in the order of keys sorted.
        
        encoded_tensor_list = []
        for key in sorted(observations.keys()):
             encoded_tensor_list.append(observations[key])
        
        x = torch.cat(encoded_tensor_list, dim=1)
        
        x = self.initial_projection(x)
        x = self.initial_activation(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        x = self.final_projection(x)
        return self.final_activation(x)

