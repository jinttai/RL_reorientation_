import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

from config.configs import Config
from src.env.space_robot_env import SpaceRobotEnv
from src.models.custom_policy import ResidualFeatureExtractor
import os

model_path = os.path.join(os.path.dirname(__file__), 'assets', 'spacerobot_cjt.xml')

class LoggingCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.final_orientation_errors = []
        self.final_joint_errors = []
        self.episode_count = 0

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
                
                if self.episode_count % 10 == 0:
                    self._log_summary()
                    
        return True

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

class CurriculumCallback(BaseCallback):
    def __init__(self, trigger_step: int, verbose: int = 0):
        super().__init__(verbose)
        self.trigger_step = trigger_step
        self.phase_switched = False
        
        # Calculate curriculum level step size (trigger_step divided by 4)
        self.level_step_size = trigger_step // 4
        self.current_level = 1

    def _on_step(self) -> bool:
        # Phase 1 Curriculum: Update level every quarter of trigger_step
        if not self.phase_switched:
            # We want levels 1, 2, 3, 4. 
            # Level increases at 1/4, 2/4, 3/4 of trigger_step
            # current_level starts at 1.
            # Next level at: level * step_size
            
            next_level_threshold = self.current_level * self.level_step_size
            
            if self.num_timesteps >= next_level_threshold and self.current_level < 4:
                self.current_level += 1
                self.training_env.set_attr("level_phase1", self.current_level)
                if self.verbose > 0:
                    print(f"\n[Curriculum] Phase 1 Level increased to {self.current_level} at step {self.num_timesteps}!")

        # Phase 2 Switch
        if self.num_timesteps >= self.trigger_step and not self.phase_switched:
            # Switch phase from 1 to 2
            # Access the underlying environment(s) via set_attr
            self.training_env.set_attr("phase", 2)
            
            if self.verbose > 0:
                print(f"\n[Curriculum] Switched to Phase 2 at step {self.num_timesteps}!")
            
            self.phase_switched = True
            
        return True

def main():
    env = SpaceRobotEnv(model_path=model_path)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=Config().MAX_EPISODE_STEPS)
    env = Monitor(env)

    # Policy kwargs should not include the callback.
    # If you wanted a custom feature extractor, it must inherit from BaseFeaturesExtractor.
    # We use the custom ResidualFeatureExtractor.
    policy_kwargs = {
        "features_extractor_class": ResidualFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": Config().FEATURES_DIM},
        "net_arch": {"pi": [256, 256], "qf": [256, 256]}
    }

    model = DDPG(
        "MultiInputPolicy",  # Required for Dict observation spaces (HER)
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=256,
        train_freq=(1, "episode"),
        gradient_steps=-1,
        action_noise=NormalActionNoise(mean=np.zeros(Config().ACTION_DIM), sigma=0.1 * np.ones(Config().ACTION_DIM)), 
        verbose=1,
    )

    callbacks = CallbackList([CurriculumCallback(trigger_step=10000), LoggingCallback()])
    model.learn(total_timesteps=100000, callback=callbacks)

    model.save("spacerobot_cjt_ddpg")
    env.close()




if __name__ == "__main__":
    main()
