import numpy as np
import gym
import time
from goldilock_env import TaskEnvironment

""" Step-1: build the environment and the basic evaluation/testing functions """

env = TaskEnvironment(max_steps=50)
env.reset()
#env.render()

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
r_sum = run_episode(render=False)
print(r_sum)

def evaluate_policy(pi=random_policy, episodes=10):
    # after running several episodes, return the avg reward per episode
    rewards=[]
    for _ in range(episodes):
        r_sum = run_episode(pi)
        rewards.append(r_sum)
    return np.mean(rewards)


r_avg = evaluate_policy(random_policy)
print(r_avg)

""" Step-2: Build a NN model for PolicyNet
- layers(dimensions): input(4)-->dense(128)-->dense(128)-->output(2)
- output should be followed by softmax-activation to generate normalized action-probability  
"""

import tensorflow as tf
state = x = tf.keras.Input(shape=env.observation_space.shape)
x = tf.keras.layers.Dense(128, "relu")(x)
x = tf.keras.layers.Dense(128, "relu")(x)
y = tf.keras.layers.Dense(3, tf.nn.softmax)(x)
PolicyNet = tf.keras.Model(inputs=state, outputs=y)

state = x = tf.keras.Input(shape=env.observation_space.shape)
x = tf.keras.layers.Dense(128, "relu")(x)
x = tf.keras.layers.Dense(128, "relu")(x)
y = tf.keras.layers.Dense(1)(x)
ValueNet = tf.keras.Model(inputs=state, outputs=y)



opt = tf.keras.optimizers.Adam(1e-4)
opt_v = tf.keras.optimizers.Adam(1e-3)

@tf.function
def optimize(states, actions, advantages): # optimize so that the MSE error btn Q(s,a) and targets is minimized
    with tf.GradientTape() as tape:
        # obtain pi(s): [N,2] probability for all the actions
        pi_s = PolicyNet(states) #[N,2]


        # obtain pi(s,a): [N,] probability for the chosen action
        # row_idx: 0 to N-1, col_idx: a
        row_idx = tf.range(0, tf.shape(states)[0])
        col_idx = actions
        row_col_idx = tf.stack([row_idx, col_idx], axis=-1)
        # use tf.gather_nd to select the chosen action-probability
        pi_sa = tf.gather_nd(params=pi_s, indices=row_col_idx) #[N] probabilities

        # define loss as - log pi(s,a) * returns
        loss = - tf.math.log(pi_sa) * advantages
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, PolicyNet.trainable_variables)
    opt.apply_gradients(zip(grads, PolicyNet.trainable_variables))

@tf.function
def optimize_v(states, targets): # optimize so that the MSE error btn Q(s,a) and targets is minimized
    with tf.GradientTape() as tape:
        # obtain v(s): [N,1]
        v_s = ValueNet(states) #[N,1]

        # define loss as - log pi(s,a) * returns
        loss = tf.square(v_s - targets)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, ValueNet.trainable_variables)
    opt_v.apply_gradients(zip(grads, ValueNet.trainable_variables))


@tf.function
def tf_policy(state): # returns the policy (action-probabilities) using the current policynet
    # print(state)
    state = tf.expand_dims(state,axis=0)
    return PolicyNet(state)

def policy(state): # returns an action based on pi(state)
    # get the policy using tf_policy and convert to numpy
    pi_s = tf_policy(state).numpy()[0] #[2]
    # choose the optimal action based on the policy
    action = np.random.choice(np.arange(0, env.action_space.n), p=pi_s)
    return action

@tf.function
def tf_value(states):
    # print(ValueNet(states))
    return ValueNet(states)

def value(states):
    # print(tf_value(states).numpy)
    return np.squeeze(tf_value(states).numpy())

r_avg = evaluate_policy(pi=policy)
print(r_avg)

""" Step-3: start exploring and training """

gamma = 0.99
max_episodes = 10000

rewards_history = []

for episode in range(max_episodes):
    # perform an episode using the current policy,
    # and store each timestep (transition info) to S,A,R
    # (we do not need S' for reinforce)
    # if episode terminates before completing the max_steps, give r=-100
    states, actions, rewards = [], [], []
    s = env.reset()
    done =False
    step = 0
    N = 0
    while not done:
        a = policy(s)
        ns, r, done, _ = env.step(a)
        step += 1
        N += 1
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = ns

        if N%5 ==0 or done == True:
            states.append(ns)
            states = np.array(states).astype(np.float32) #6 states, s = state[:-1], s' = [1:]
            actions = np.array(actions).astype(np.int32) #5 actions
            rewards = np.array(rewards).astype(np.float32) #5 reward

            #compute targets and advantages
            #targets = r + gamma*v(s')
            values = value(states) #6 values for 6 states
            # print(f'test:{values}')
            targets = rewards + gamma * values[1:]

            if rewards[-1] == -100.0:
                targets[-1] = rewards[-1]

            #compute the advantages
            # advantages = targets - V(s)
            advantages = targets - values[:-1]

            #optimizes
            optimize(states[:-1], actions, advantages)
            optimize_v(states[:-1], targets)

            #reinitialize
            N = 0
            states, actions, rewards = [], [], []


    # after each 50 episodes
    if episode % 50 == 0:
        # evaluate the greedy policy; add the avg_reward to the rewards_history
        r_avg = evaluate_policy(pi=policy)
        print(episode, r_avg)
        rewards_history.append(r_avg)

r_sum = run_episode(pi=policy, render=True)
print(r_sum)