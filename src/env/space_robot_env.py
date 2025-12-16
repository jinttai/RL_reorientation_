import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple, MultiBinary
import os 
from scipy.spatial.transform import Rotation as R

from config.configs import Config

class SpaceRobotEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, model_path, frame_skip=1):
        utils.EzPickle.__init__(self, model_path)

        self._target_base_euler = self._sample_base_euler()
        self._target_joint_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.current_step = 0
        self.prev_action = np.zeros(Config().ACTION_DIM)  # For action smoothing

        self.n_joints = 6

        # Observation: state + target (for non-HER setup)
        # state: base_euler(3) + joint_angles(6) + base_ang_vel(3) + joint_vels(6) + timestep(1) = 19
        # target: target_base_euler(3) + target_joint_angles(6) = 9
        # total: 28
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(Config().OBS_DIM + Config().GOAL_DIM,)
        )

        MujocoEnv.__init__(self, model_path, frame_skip=frame_skip, observation_space=self.observation_space)

    def _quat_mujoco_to_euler_zyx(self, quat_wxyz):
        """MuJoCo [w,x,y,z] -> scipy quaternion [x,y,z,w] -> euler ZYX [z,y,x]"""
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        euler_zyx = R.from_quat(quat_xyzw).as_euler('zyx', degrees=False)
        return euler_zyx
    
    def _sample_base_euler(self):
        # Sample random euler angles (ZYX order)
        # ZYX means: rotate around Z, then Y, then X
        # Sample random rotation up to 40 degrees
        max_angle_rad = 40.0 * np.pi / 180.0
        
        # Random euler angles in ZYX order
        euler_z = np.random.uniform(-max_angle_rad, max_angle_rad)
        euler_y = np.random.uniform(-max_angle_rad, max_angle_rad)
        euler_x = np.random.uniform(-max_angle_rad, max_angle_rad)
        
        return np.array([euler_z, euler_y, euler_x])

    def _get_obs(self):
        if self.data is None:
            print("Data is None")
            return np.zeros(Config().OBS_DIM + Config().GOAL_DIM)
        
        # Get quaternion from MuJoCo and convert to euler ZYX
        base_quat = self.data.qpos[3:7].copy()  # MuJoCo format [w,x,y,z]
        base_euler = self._quat_mujoco_to_euler_zyx(base_quat)
        
        joint_angles = self.data.qpos[7:].copy()
        base_ang_vel = self.data.qvel[3:6].copy()
        joint_vels = self.data.qvel[6:].copy()
        
        # Normalize timestep to [0, 1] range
        normalized_timestep = self.current_step / Config().MAX_EPISODE_STEPS

        # Current state
        state = np.concatenate([
            base_euler,
            joint_angles,
            base_ang_vel,
            joint_vels,
            np.array([normalized_timestep])
        ])

        # Target state
        target = np.concatenate([
            self._target_base_euler,
            self._target_joint_angles
        ])

        # Concatenate state and target
        obs = np.concatenate([state, target])
        return obs

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        self._target_base_euler = self._sample_base_euler()
        self._target_joint_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.current_step = 0
        self.prev_action = np.zeros(Config().ACTION_DIM)  # Reset previous action
        
        self.set_state(qpos, qvel)
        
        # 초기 상태 확인: reset 직후 done이 되지 않도록 보장
        obs = self._get_obs()
        # obs: [state(19), target(9)]
        current_base_euler = obs[0:3]  # base_euler in state
        target_base_euler = obs[19:22]  # target_base_euler in target
        initial_euler_diff = np.linalg.norm(current_base_euler - target_base_euler)
        
        # 초기 상태가 이미 목표와 너무 가까우면 목표를 다시 샘플링
        max_retries = 10
        retry_count = 0
        while initial_euler_diff < Config().ORIENTATION_DONE_THRESHOLD * 2 and retry_count < max_retries:
            self._target_base_euler = self._sample_base_euler()
            obs = self._get_obs()
            current_base_euler = obs[0:3]
            target_base_euler = obs[19:22]
            initial_euler_diff = np.linalg.norm(current_base_euler - target_base_euler)
            retry_count += 1
        
        return obs

    def step(self, action):
        self.current_step += 1

        # CTC (Computed Torque Control): action is desired joint velocity
        desired_joint_velocity = action * Config().MAX_JOINT_VELOCITY
        torque = self._compute_torque_ctc(desired_joint_velocity)
        self.do_simulation(torque, self.frame_skip)

        obs = self._get_obs()
        
        # Extract state and target from observation
        # obs: [state(19), target(9)]
        state = obs[:Config().OBS_DIM]
        target = obs[Config().OBS_DIM:]
        
        # Current state components
        current_base_euler = state[0:3]
        current_joint_angles = state[3:9]
        # timestep is at state[18]
        
        # Target components
        target_base_euler = target[0:3]
        target_joint_angles = target[3:9]
        
        # Calculate errors for logging (euler angle difference)
        euler_diff = np.linalg.norm(current_base_euler - target_base_euler)
        joint_diff = np.sum(np.abs(current_joint_angles - target_joint_angles))

        info = {
            'orientation_error': euler_diff,
            'joint_error': joint_diff,
            'current_step': self.current_step,
            'prev_action': self.prev_action.copy()
        }

        reward = self.compute_reward(state, target, info, action)
        
        # Update previous action for next step
        self.prev_action = action.copy()
        
        # done 계산: 최소 1 스텝은 실행되도록 보장
        terminated = False
        if self.current_step > 1:
            terminated = self._compute_done(state, target, info)
    
        truncated = False

        return obs, reward, terminated, truncated, info

    def compute_reward(self, state, target, info, action):
        # state: [base_euler(3), joint_angles(6), base_ang_vel(3), joint_vels(6), timestep(1)]
        # target: [target_base_euler(3), target_joint_angles(6)]
        
        current_base_euler = state[0:3]
        target_base_euler = target[0:3]
        
        # Calculate orientation error (L2 norm of euler angle difference)
        euler_diff = np.linalg.norm(current_base_euler - target_base_euler)
        joint_vel = np.linalg.norm(state[9:15])
        
        if euler_diff > 0.5 :
            reward = -((euler_diff*2)**2) * Config().ORIENTATION_REWARD_WEIGHT
        else:
            reward = -(euler_diff*2) * Config().ORIENTATION_REWARD_WEIGHT

        # Joint velocity penalty: only apply after 50 steps
        current_step = info.get('current_step', 0)
        if current_step > 50:
            reward += -(joint_vel**2) * Config().JOINT_VELS_REWARD_WEIGHT
        
        # Action smoothing reward: penalize large changes in action
        if 'prev_action' in info:
            action_diff = np.linalg.norm(action - info['prev_action'])
            reward += -(action_diff**2) * Config().ACTION_SMOOTHING_REWARD_WEIGHT
        
        if euler_diff < Config().ORIENTATION_DONE_THRESHOLD:
            reward += 1.0
        return reward

    def _compute_done(self, state, target, info):
        # state: [base_euler(3), joint_angles(6), base_ang_vel(3), joint_vels(6), timestep(1)]
        # target: [target_base_euler(3), target_joint_angles(6)]
        
        current_base_euler = state[0:3]
        target_base_euler = target[0:3]
        
        # Calculate orientation error (L2 norm of euler angle difference)
        euler_diff = np.linalg.norm(current_base_euler - target_base_euler)
        
        done = euler_diff < Config().ORIENTATION_DONE_THRESHOLD

        return done

    def _compute_torque_ctc(self, desired_joint_velocity):
        """
        CTC (Computed Torque Control): Compute torque to achieve desired joint velocity
        τ = M(q)(q̈_des) + C(q, q̇)q̇ + g(q)
        Simplified version: PD control with desired velocity
        """
        current_joint_velocity = self.data.qvel[6:].copy()
        current_joint_angles = self.data.qpos[7:].copy()
        
        # Velocity error
        velocity_error = desired_joint_velocity - current_joint_velocity
        
        # PD control: τ = KP * velocity_error + KD * (desired_acceleration)
        # For CTC, we want to track desired velocity, so we use PD control
        # τ = KP * (q̇_des - q̇) + KD * (q̈_des)
        # Simplified: τ = KP * velocity_error
        torque = Config().KP * velocity_error
        
        # Add damping term
        torque += Config().KD * (desired_joint_velocity - current_joint_velocity)
        
        # Clip torque
        torque = np.clip(torque, -Config().MAX_TORQUE, Config().MAX_TORQUE)

        return torque