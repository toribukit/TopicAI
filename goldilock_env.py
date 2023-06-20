import numpy as np
import gym
import time
from gym import spaces

class TaskEnvironment(gym.Env):
    def __init__(self, max_steps=200):
        self.difficulty_level = 0.0  # Initial difficulty level (ranging from 0 to 1)
        self.max_steps = max_steps  # Maximum number of steps before termination
        self.steps_taken = 0  # Number of steps taken in the environment

        # Define the action space and observation space
        self.action_space = spaces.Discrete(3)  # Three possible actions: 0 for decreasing difficulty, 1 for increasing difficulty, 2 for no change
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)  # Continuous observation space for difficulty level

    def reset(self):
        self.difficulty_level = 0.0  # Reset difficulty level to the initial value
        self.steps_taken = 0  # Reset step count
        return np.array([self.difficulty_level], dtype=np.float32)

    def step(self, action):
        if action == 0:  # Decrease difficulty level
            self.difficulty_level = max(0.0, self.difficulty_level - 0.1)
        elif action == 1:  # Increase difficulty level
            self.difficulty_level = min(1.0, self.difficulty_level + 0.1)

        reward = self._calculate_reward(self.difficulty_level)
        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps

        return np.array([self.difficulty_level], dtype=np.float32), reward, done, {}

    def _calculate_reward(self, difficulty_level):
        peak = 0.5  # Peak difficulty level for the reward distribution
        std_dev = 0.2  # Standard deviation of the Gaussian distribution

        reward = np.exp(-0.5 * ((difficulty_level - peak) / std_dev) ** 2)
        return reward

    def render(self):
        print(f"Difficulty Level: {self.difficulty_level}")

# env = TaskEnvironment(max_steps=50)
# obs = env.reset()
# done = False

class TaskEnvironmentDynamic(gym.Env):
    def __init__(self, max_steps=200):
        self.difficulty_level = 0.0  # Initial difficulty level (ranging from 0 to 1)
        self.max_steps = max_steps  # Maximum number of steps before termination
        self.steps_taken = 0  # Number of steps taken in the environment

        # Define the action space and observation space
        self.action_space = spaces.Discrete(3)  # Three possible actions: 0 for decreasing difficulty, 1 for increasing difficulty, 2 for no change
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)  # Continuous observation space for difficulty level and steps taken

    def reset(self):
        self.difficulty_level = 0.0  # Reset difficulty level to the initial value
        self.steps_taken = 0  # Reset step count
        return np.array([self.difficulty_level, self.steps_taken], dtype=np.float32)

    def step(self, action):
        if action == 0:  # Decrease difficulty level
            self.difficulty_level = max(0.0, self.difficulty_level - 0.1)
        elif action == 1:  # Increase difficulty level
            self.difficulty_level = min(1.0, self.difficulty_level + 0.1)

        reward = self._calculate_reward(self.difficulty_level)
        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps

        return np.array([self.difficulty_level, self.steps_taken], dtype=np.float32), reward, done, {}

    def _calculate_reward(self, difficulty_level):
        peak_start = 0.3  # Starting difficulty level for the reward peak
        peak_end = 0.7  # Ending difficulty level for the reward peak
        std_dev = 0.2  # Standard deviation of the Gaussian distribution

        peak = peak_start + (peak_end - peak_start) * (self.steps_taken / self.max_steps)  # Calculate dynamic peak based on steps_taken
        reward = np.exp(-0.5 * ((difficulty_level - peak) / std_dev) ** 2)
        return reward

    def render(self):
        print(f"Difficulty Level: {self.difficulty_level}, Steps Taken: {self.steps_taken}")