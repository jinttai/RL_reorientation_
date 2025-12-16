import csv
import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

import os
from typing import Optional

from config.configs import Config
from src.env.space_robot_env import SpaceRobotEnv
from src.models.custom_policy import SimpleMLPFeatureExtractor

model_path = os.path.join(os.path.dirname(__file__), 'assets', 'spacerobot_cjt.xml')

def linear_schedule(initial_value: float, final_value: float):
    """
    Linear schedule for SB3.
    progress_remaining goes from 1 (start) to 0 (end).
    """
    initial_value = float(initial_value)
    final_value = float(final_value)

    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return func

class LoggingCallback(BaseCallback):
    def __init__(self, csv_path: Optional[str] = None, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.final_orientation_errors = []
        self.final_joint_errors = []
        self.episode_count = 0
        self.csv_path = csv_path or os.path.join(os.path.dirname(__file__), "training_metrics.csv")
        self._csv_initialized = False

    def _on_step(self) -> bool:
        # Check for end of episode
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        for i, done in enumerate(dones):
            if done:
                self.episode_count += 1
                info = infos[i]
                
                # Monitor wrapper adds 'episode' key with 'r' (reward) and 'l' (length)
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                
                # Custom metrics from SpaceRobotEnv
                if 'orientation_error' in info:
                    self.final_orientation_errors.append(info['orientation_error'])
                
                if 'joint_error' in info:
                    self.final_joint_errors.append(info['joint_error'])

                episode_reward = info['episode']['r'] if 'episode' in info else np.nan
                episode_length = info['episode']['l'] if 'episode' in info else np.nan
                orientation_error = info.get('orientation_error', np.nan)
                joint_error = info.get('joint_error', np.nan)

                self._write_csv_row([
                    self.episode_count,
                    self.num_timesteps,
                    episode_reward,
                    episode_length,
                    orientation_error,
                    joint_error,
                ])
                
                if self.episode_count % 10 == 0:
                    self._log_summary()
                    
        return True

    def _ensure_csv(self):
        if self._csv_initialized:
            return
        file_exists = os.path.exists(self.csv_path)
        file_empty = not file_exists or os.path.getsize(self.csv_path) == 0
        if file_empty:
            with open(self.csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "timesteps",
                    "episode_reward",
                    "episode_length",
                    "orientation_error",
                    "joint_error",
                ])
        self._csv_initialized = True

    def _write_csv_row(self, row_values):
        self._ensure_csv()
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row_values)

    def _log_summary(self):
        mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        mean_ori_error = np.mean(self.final_orientation_errors) if self.final_orientation_errors else 0.0
        mean_joint_error = np.mean(self.final_joint_errors) if self.final_joint_errors else 0.0
        
        print(f"\n[Episode {self.episode_count}] (Last 10 Episodes) "
              f"Mean Reward: {mean_reward:.2f}, "
              f"Mean Final Ori Error: {mean_ori_error:.4f}, "
              f"Mean Final Joint Error: {mean_joint_error:.4f}")
        
        self.episode_rewards = []
        self.final_orientation_errors = []
        self.final_joint_errors = []

def main():
    cfg = Config()

    def make_env():
        e = SpaceRobotEnv(model_path=model_path)
        e = gym.wrappers.TimeLimit(e, max_episode_steps=cfg.MAX_EPISODE_STEPS)
        e = Monitor(e)
        return e

    env = DummyVecEnv([make_env])
    if cfg.NORM_OBS or cfg.NORM_REWARD:
        env = VecNormalize(
            env,
            norm_obs=cfg.NORM_OBS,
            norm_reward=cfg.NORM_REWARD,
            clip_obs=cfg.CLIP_OBS,
        )

    # Policy kwargs should not include the callback.
    # If you wanted a custom feature extractor, it must inherit from BaseFeaturesExtractor.
    # We use the custom ResidualFeatureExtractor.
    policy_kwargs = {
        "features_extractor_class": SimpleMLPFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": Config().FEATURES_DIM},
        "net_arch": {"pi": [256, 256], "qf": [256, 256]},
    }

    model = TD3(
        "MlpPolicy",  # Standard MLP policy for Box observation space
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(cfg.LR_INITIAL, cfg.LR_FINAL),
        gamma=0.99,
        batch_size=512,
        train_freq=(1, "episode"),
        gradient_steps=-1,
        action_noise=NormalActionNoise(mean=np.zeros(cfg.ACTION_DIM), sigma=0.1 * np.ones(cfg.ACTION_DIM)),
        policy_delay=2,  # TD3 specific: delay policy updates
        target_policy_noise=0.2,  # TD3 specific: noise added to target policy
        target_noise_clip=0.5,  # TD3 specific: clip target policy noise
        verbose=1,
    )

    callbacks = CallbackList([LoggingCallback()])
    model.learn(total_timesteps=cfg.TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True, log_interval=10)

    model.save("spacerobot_cjt_td3")
    # Save VecNormalize running stats so you can load & evaluate consistently later.
    if isinstance(env, VecNormalize):
        env.save("spacerobot_cjt_td3_vecnormalize.pkl")
    env.close()




if __name__ == "__main__":
    main()
