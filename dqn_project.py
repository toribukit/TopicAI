import numpy as np
import gym
import time
from goldilock_env import TaskEnvironment

""" Step-1: build the environment and the basic evaluation/testing functions """

env = TaskEnvironment(max_steps=50)
env.reset()
# env.render()

print(env.observation_space)
print(env.action_space)


def random_policy(state):
    return env.action_space.sample()


def run_episode(pi=random_policy, render=False):
    r_sum = 0
    done = False
    s = env.reset()

    while not done:
        a = pi(s)
        ns, r, done, _ = env.step(a)
        r_sum += r
        s = ns

        if render:
            env.render()
            time.sleep(0.01)

    return r_sum


# test run_episode
# run_episode(random_policy, render=True)

def evaluate_policy(pi=random_policy, episodes=10):
    # after running several episodes, return the avg reward per episode
    rewards = []
    for _ in range(episodes):
        r_sum = run_episode(pi)
        rewards.append(r_sum)
    return np.mean(rewards)


r_avg = evaluate_policy(random_policy)
print(r_avg)

""" Step-2: Build two identical NN models for QNET and QTARGET
- layers(dimensions): input(4)-->dense(128)-->dense(128)-->output(2)  
"""

import tensorflow as tf

state = x = tf.keras.Input(shape=env.observation_space.shape)
x = tf.keras.layers.Dense(units=128, activation='relu')(x)
x = tf.keras.layers.Dense(units=128, activation='relu')(x)
q = x = tf.keras.layers.Dense(units=3)(x)
qnet = tf.keras.Model(inputs=state, outputs=q)

# ...
state = x = tf.keras.Input(shape=env.observation_space.shape)
x = tf.keras.layers.Dense(units=128, activation='relu')(x)
x = tf.keras.layers.Dense(units=128, activation='relu')(x)
q = x = tf.keras.layers.Dense(units=3)(x)
qtarget = tf.keras.Model(inputs=state, outputs=q)

opt = tf.keras.optimizers.Adam(1e-4)


@tf.function
def optimize(states, actions, targets):  # optimize so that the MSE error btn Q(s,a) and targets is minimized
    with tf.GradientTape() as tape:
        # obtain Q(s): [N,2] value for all the actions
        qs = qnet(states)
        # obtain Q(s,a): [N,] value for the chosen action
        col_idx = actions
        n = tf.shape(qs)[0]
        row_idx = tf.range(n)
        sa_idx = tf.stack([row_idx, col_idx], axis=-1)
        qsa = tf.gather_nd(params=qs, indices=sa_idx)
        # calculate MSE between Q(s,a) and targets
        mse = tf.reduce_mean(tf.square(targets - qsa))

    grads = tape.gradient(mse, qnet.trainable_variables)
    opt.apply_gradients(zip(grads, qnet.trainable_variables))


@tf.function
def q_policy(state):  # returns the greedy policy for the current qnet
    print(state)
    state = tf.expand_dims(state, axis=0)  # add batch dims [1, ...]
    qs = qnet(state)[0]  # [2] values
    return tf.argmax(qs)  # index of max value


def greedy_policy(state):  # returns the greedy policy for the current qnet
    return q_policy(state).numpy()


r_avg = evaluate_policy(pi=greedy_policy)

action_prob = np.zeros(shape=[env.action_space.n])


def epsilon_policy(state, epsilon=0.3):
    m = env.action_space.n
    action_prob[:] = epsilon / m
    a = greedy_policy(state)
    action_prob[a] += (1 - epsilon)
    return np.random.choice(np.arange(0, env.action_space.n), p=action_prob)


from collections import deque


class Replay_Memory:  # stores transitions, gives random batches
    def __init__(self, maxlen=10000, batch_size=64):
        self.batch_size = batch_size
        self.buffers = []
        # create four buffers (queues) for storing states, actions, rewards, next_states
        for _ in range(4):
            buffers = deque(maxlen=maxlen)
            self.buffers.append(buffers)

    def size(self):
        return len(self.buffers[0])

    def append(self, timestep):  # timestep: (s,a,r,ns)
        # append timestep info to corresponding buffers
        for i in range(4):
            self.buffers[i].append(timestep[i])

    def get_batches(self):  # returns random samples of transitions
        n = self.size()
        # obtain shuffled indices from 0 to n-1
        shuffle_idx = np.random.permutation(n)
        # obtain batch indices
        batch_idx = shuffle_idx[0:self.batch_size]

        samples = []
        for i in range(4):
            batch = np.array(self.buffers[i])[batch_idx]
            samples.append(batch)
        return samples


replay_mem = Replay_Memory()


@tf.function
def qtarget_pred(states):  # [batch,2]
    return qtarget(states)


gamma = 0.99
max_episodes = 10000
epsilon = np.linspace(0.9, 0.0, max_episodes)  # why this instead of epsilon=0.3?

rewards_history = []

qtarget.set_weights(qnet.get_weights())

for episode in range(max_episodes):
    # perform an episode using epsilon-greedy policy,
    # and store each timestep (transition info) to the replay memory
    s = env.reset()
    done = False
    step = 0
    while not done:
        a = epsilon_policy(s, epsilon[episode])
        ns, r, done, _ = env.step(a)
        step += 1

        replay_mem.append([s, a, r, ns])
        s = ns

    if replay_mem.size() < 100:  # why this?
        continue

    # get a random batch : (states, actions, rewards, next_states) from replay memory
    states, actions, rewards, next_states = replay_mem.get_batches()

    # compute the targets: r + gamma*max(Q(s'));
    # reminder: here, Q is not QNET but QTARGET
    q_ns = qtarget_pred(next_states).numpy()  # [N,2]
    max_q = np.max(q_ns, axis=1)
    max_q[rewards < 0] = 0
    targets = rewards + gamma * max_q

    # optimize to move towards the targets
    # print(states[0])
    optimize(states, actions, np.float32(targets))

    # after each 50 episodes
    if episode % 50 == 0:
        # evaluate the greedy policy; add the avg_reward to the rewards_history
        r_avg = evaluate_policy(pi=greedy_policy)
        print(episode, r_avg)
        rewards_history.append(r_avg)

    # after each 50 episodes, sync QTARGET to QNET
    if episode % 50 == 0:
        qtarget.set_weights(qnet.get_weights())

r_sum = run_episode(pi=greedy_policy, render=True)
print(r_sum)