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
#
# while not done:
#     action = ...  # Choose an action based on your RL algorithm
#     next_obs, reward, done, _ = env.step(action)
#     env.render()
#
# env.close()

# env = TaskEnvironment(max_steps=50)
# print(env.action_space)  # Print the action space
# print(env.observation_space)  # Print the observation space
#
# def random_policy(state):
#     return env.action_space.sample()
#
#
# def run_episode(pi=random_policy, render=False):
#     r_sum = 0
#     done = False
#     s = env.reset()
#
#     while not done:
#         a = pi(s)
#         # a = 1
#         ns, r, done, _ = env.step(a)
#         r_sum += r
#         s = ns
#
#         if render:
#             env.render()
#             time.sleep(0.01)
#
#     return r_sum
#
# def evaluate_policy(pi=random_policy, episodes=1):
#     # after running several episodes, return the avg reward per episode
#     rewards=[]
#     for _ in range(episodes):
#         r_sum = run_episode(pi)
#         rewards.append(r_sum)
#     return np.mean(rewards)
#
# r_avg = evaluate_policy(random_policy)
# print(r_avg)
