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
        self.difficulty_level = 0.0
        self.max_steps = max_steps
        self.steps_taken = 0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self):
        self.difficulty_level = 0.0
        self.steps_taken = 0
        return np.array([self.difficulty_level, self.steps_taken], dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.difficulty_level = max(0.0, self.difficulty_level - 0.1)
        elif action == 1:
            self.difficulty_level = min(1.0, self.difficulty_level + 0.1)

        reward = self._calculate_reward(self.difficulty_level, self.steps_taken)
        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps

        return np.array([self.difficulty_level, self.steps_taken], dtype=np.float32), reward, done, {}

    def _calculate_reward(self, difficulty_level, steps_taken):
        base_peak = 0.3
        max_peak = 0.7
        peak_increment = 0.1

        current_peak = min(base_peak + (steps_taken // 10) * peak_increment, max_peak)

        std_dev = 0.2

        reward = np.exp(-0.5 * ((difficulty_level - current_peak) / std_dev) ** 2)
        return reward

    def render(self):
        print(f"Difficulty Level: {self.difficulty_level}")



# env = TaskEnvironmentDynamic()
#
# obs = env.reset()
# env.render()
#
# for _ in range(20):
#     action = env.action_space.sample()
#     obs, reward, done, _ = env.step(action)
#
#     env.render()
#     print(f"Action: {action}, Reward: {reward}, Done: {done}, state: {obs.shape}")
#
#     if done:
#         break

class TaskEnvironmentTwoPeak(gym.Env):
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
        peak = 0.7  # Peak difficulty level for the regular Gaussian distribution
        std_dev = 0.2  # Standard deviation of the regular Gaussian distribution

        if difficulty_level < peak:
            upside_down_peak = 1.0 - peak  # Peak difficulty level for the upside-down Gaussian distribution
            upside_down_std_dev = std_dev  # Standard deviation of the upside-down Gaussian distribution

            reward = np.exp(-0.5 * ((difficulty_level - upside_down_peak) / upside_down_std_dev) ** 2)
            reward = max(0.0, reward - 0.3)
        else:
            reward = np.exp(-0.5 * ((difficulty_level - peak) / std_dev) ** 2)

        reward = np.array(reward) ** 2

        return reward

    def render(self):
        print(f"Difficulty Level: {self.difficulty_level}")

# import matplotlib.pyplot as plt
#
# def calculate_reward(difficulty_level):
#     peak = 0.7  # Peak difficulty level for the regular Gaussian distribution
#     std_dev = 0.2  # Standard deviation of the regular Gaussian distribution
#
#     if difficulty_level < peak:
#         upside_down_peak = 1.0 - peak  # Peak difficulty level for the upside-down Gaussian distribution
#         upside_down_std_dev = std_dev  # Standard deviation of the upside-down Gaussian distribution
#
#         reward = np.exp(-0.5 * ((difficulty_level - upside_down_peak) / upside_down_std_dev) ** 2)
#         reward = max(0.0, reward - 0.3)
#     else:
#         reward = np.exp(-0.5 * ((difficulty_level - peak) / std_dev) ** 2)
#
#     reward = np.array(reward) ** 2
#
#     return reward
#
#
# # Generate data for plotting
# difficulty_levels = np.linspace(0.0, 1.0, 100)
# rewards = [calculate_reward(level) for level in difficulty_levels]


# Plotting
# plt.plot(difficulty_levels, rewards)
# plt.xlabel('Difficulty Level')
# plt.ylabel('Reward')
# plt.title('Reward Function')
# plt.grid(True)
# plt.show()
