import numpy as np

class Config:
    def __init__(self):
        self.OBS_DIM = 19
        self.GOAL_DIM = 10
        self.ACTION_DIM = 6
        self.MAX_EPISODE_STEPS = 100
        self.MAX_EPISODE_LENGTH = 100

        self.KP = 10.0
        self.KD = 1.0
        self.MAX_TORQUE = 10.0
        self.MAX_JOINT_VELOCITY = 10.0

        self.CONTROL_DT = 0.1

        self.ORIENTATION_REWARD_WEIGHT = 1.0
        self.JOINT_REWARD_WEIGHT = 1.0
        self.ORIENTATION_DONE_THRESHOLD = 0.01
        self.JOINT_DONE_THRESHOLD = 0.01
        
        self.FEATURES_DIM = 256
        self.HIDDEN_SIZE = 256