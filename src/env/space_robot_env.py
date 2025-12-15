import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple, MultiBinary
import os 
from scipy.spatial.transform import Rotation as R

from config.configs import Config

class SpaceRobotEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, model_path, frame_skip=10):
        utils.EzPickle.__init__(self, model_path)

        self.phase = 1
        self.level_phase1 = 1

        self._target_base_quat = self._sample_base_quat()
        self._target_joint_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.n_joints = 6

        self.observation_space = Dict({
            'observation': Box(low=-np.inf, high=np.inf, shape=(Config().OBS_DIM,)),
            'achieved_goal': Box(low=-np.inf, high=np.inf, shape=(Config().GOAL_DIM,)),
            'desired_goal': Box(low=-np.inf, high=np.inf, shape=(Config().GOAL_DIM,)),
        })

        MujocoEnv.__init__(self, model_path, frame_skip=frame_skip, observation_space=self.observation_space)
    
    def _sample_base_quat(self):
        # random rotation direction for specific rotation angle
        random_direction = np.random.randn(3)
        random_direction /= np.linalg.norm(random_direction)

        # sample rotation angle between 0 and level_phase1 * 10 degrees
        rotation_angle = np.random.uniform(0, self.level_phase1 * 10 * np.pi / 180.0)
        rotation_axis = np.cross(random_direction, np.array([0, 0, 1]))
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_quat = R.from_rotvec(rotation_angle * rotation_axis).as_quat()
        return rotation_quat

    def _get_obs(self):
        if self.data is None:
            print("Data is None")
            return{
                'observation': np.zeros(Config().OBS_DIM),
                'achieved_goal': np.zeros(Config().GOAL_DIM),
                'desired_goal': np.zeros(Config().GOAL_DIM),
            }
        base_quat = self.data.qpos[3:7].copy()
        joint_angles = self.data.qpos[7:].copy()
        base_ang_vel = self.data.qvel[3:6].copy()
        joint_vels = self.data.qvel[6:].copy()

        obs = np.concatenate([
            base_quat,
            joint_angles,
            base_ang_vel,
            joint_vels
        ])

        achieved_goal = np.concatenate([
            base_quat,
            joint_angles
        ])
        desired_goal = np.concatenate([
            self._target_base_quat,
            self._target_joint_angles
        ])

        return {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
        }

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        self.target_base_quat = self._sample_base_quat()
        self.target_joint_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):
        joint_velocity_command = action * Config().MAX_JOINT_VELOCITY
        torque = self._compute_torque(joint_velocity_command)
        self.do_simulation(torque, self.frame_skip)

        obs = self._get_obs()
        
        # Calculate errors for logging
        achieved_goal = obs['achieved_goal']
        desired_goal = obs['desired_goal']
        
        achieved_base_quat = achieved_goal[:4]
        desired_base_quat = desired_goal[:4]
        quat_diff = 1 - np.abs(np.dot(achieved_base_quat, desired_base_quat))
        
        achieved_joint_angles = achieved_goal[4:]
        desired_joint_angles = desired_goal[4:]
        joint_diff = np.sum(np.abs(achieved_joint_angles - desired_joint_angles))

        info = {
            'orientation_error': quat_diff,
            'joint_error': joint_diff
        }

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        done = self._compute_done(obs['achieved_goal'], obs['desired_goal'])
        
        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Handle 1D input (step) vs 2D input (replay buffer)
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
            single_input = True
        else:
            single_input = False

        achieved_base_quat = achieved_goal[:, :4]
        achieved_joint_angles = achieved_goal[:, 4:]
        
        desired_base_quat = desired_goal[:, :4]
        desired_joint_angles = desired_goal[:, 4:]
        
        # Batch dot product for quaternions
        dot_prod = np.sum(achieved_base_quat * desired_base_quat, axis=1)
        quat_diff = 1 - np.abs(dot_prod)
        
        # Batch sum for joint angles
        joint_diff = np.sum(np.abs(achieved_joint_angles - desired_joint_angles), axis=1)
        
        reward_orientation = -quat_diff * Config().ORIENTATION_REWARD_WEIGHT
        reward_orientation = np.where(quat_diff < Config().ORIENTATION_DONE_THRESHOLD, 1.0, reward_orientation)
        
        reward_joint = -joint_diff * Config().JOINT_REWARD_WEIGHT
        reward_joint = np.where(joint_diff < Config().JOINT_DONE_THRESHOLD, 1.0, reward_joint)
            

        reward = reward_orientation 

        if self.phase == 2:
            reward += reward_joint

        if single_input:
            return reward[0]
        return reward

    def _compute_done(self, achieved_goal, desired_goal):
        # achieved_goal and desired_goal are arrays
        achieved_base_quat = achieved_goal[:4]
        achieved_joint_angles = achieved_goal[4:]
        
        desired_base_quat = desired_goal[:4]
        desired_joint_angles = desired_goal[4:]
        
        quat_diff = 1 - np.abs(np.dot(achieved_base_quat, desired_base_quat))
        joint_diff = np.sum(np.abs(achieved_joint_angles - desired_joint_angles))
        
        done = quat_diff < Config().ORIENTATION_DONE_THRESHOLD
        
        if self.phase == 2:
            done = done and joint_diff < Config().JOINT_DONE_THRESHOLD

        return done

    def _compute_torque(self, action):
        torque = Config().KP * (action - self.data.qpos[7:])
        torque = np.clip(torque, -Config().MAX_TORQUE, Config().MAX_TORQUE)

        return torque