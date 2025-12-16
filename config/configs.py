import numpy as np

class Config:
    def __init__(self):
        self.OBS_DIM = 19  # base_euler(3) + joint_angles(6) + base_ang_vel(3) + joint_vels(6) + timestep(1)
        self.GOAL_DIM = 9   # target_base_euler(3) + target_joint_angles(6)
        self.ACTION_DIM = 6
        self.MAX_EPISODE_STEPS = 100
        self.MAX_EPISODE_LENGTH = 100

        self.KP = 0.1
        self.KD = 10.0
        self.MAX_TORQUE = 0.1
        self.MAX_JOINT_VELOCITY = 1.0
        self.ACTION_SMOOTHING_REWARD_WEIGHT = 0.01

        self.CONTROL_DT = 0.1

        self.ORIENTATION_REWARD_WEIGHT = 0.5
        self.JOINT_VELS_REWARD_WEIGHT = 0.1
        self.ORIENTATION_DONE_THRESHOLD = 0.05
        self.JOINT_DONE_THRESHOLD = 0.01
        
        self.FEATURES_DIM = 512
        self.HIDDEN_SIZE = 512