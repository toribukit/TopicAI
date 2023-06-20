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



opt = tf.keras.optimizers.Adam(1e-4)

@tf.function
def optimize(states, actions, returns): # optimize so that the MSE error btn Q(s,a) and targets is minimized
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
        loss = - tf.math.log(pi_sa) * returns
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, PolicyNet.trainable_variables)
    opt.apply_gradients(zip(grads, PolicyNet.trainable_variables))


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

r_avg = evaluate_policy(pi=policy)
print(r_avg)

""" Step-3: start exploring and training """

gamma = 0.99
max_episodes = 10000

# import plotter
# plotter.init()
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
    while not done:
        a = policy(s)
        ns, r, done, _ = env.step(a)
        step += 1

        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = ns




    # compute v_t (returns): discounted return at t
    returns = []
    for t in range(len(rewards)):
        future_rewards = rewards[t:]
        discounted_sum = 0
        for i, r in enumerate(future_rewards):
            discounted_sum += (gamma**i) * r
        returns.append(discounted_sum)


    # optimize towards the returns
    optimize(np.array(states), actions, np.float32(returns))

    # after each 50 episodes
    if episode % 50 == 0:
        # evaluate the greedy policy; add the avg_reward to the rewards_history
        r_avg = evaluate_policy(pi=policy)
        print(episode, r_avg)
        rewards_history.append(r_avg)
        # plotter.update_plot(rewards_history)

r_sum = run_episode(pi=policy, render=True)
print(r_sum)