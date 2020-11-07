import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append('./deepq')
sys.path.append('./utils')

import gym
import numpy as np
import pickle
from mountain_car_gravity import MountainCarEnv

from deepq.policies import MlpPolicy, LnMlpPolicy
from deepq.dqn_transfer import DQN_TRANSFER

task = 4
source = 1
max_timesteps = 100000
max_trials = 100
rewards = np.zeros(max_trials)

path_TS = f'output2/grav{source}.0/'
mname_TS = 'mc_model'

path_TD = f'output2/T{task}wT{source}/'
mname_TD = 'transfer_model'

env = MountainCarEnv(grav=task)
state = env.reset()

pkl_filename = f'{path_TD}/svm_model_T{source}.pkl'
with open(pkl_filename, 'rb') as file:
    svm_TS = pickle.load(file)

def testing():
    for i in range(max_trials):
        print(f'Test {i}')
        done = False
        state = env.reset()
        while not done:
            action, _ = model_TD.predict(state)
            state, reward, done, _ = env.step(action)
            rewards[i] += reward
    np.save(f'{path_TD}/test_rewards.npy', rewards)
    print(f'Mean: {np.mean(rewards)}')
    print(f'Std: {np.std(rewards)}')

for i in range(1,11):
    model_TD = DQN_TRANSFER(
        policy=LnMlpPolicy,
        env=env,
        sim=i,
        path_model_Tsource = f'{path_TS}/{mname_TS}_{i}',
        one_classifier = svm_TS,
        gamma=0.99,
        learning_rate=1e-3,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        double_q=False,
        prioritized_replay=False,
        verbose=1,
        tensorboard_log=path_TD,
        full_tensorboard_log=True,
        policy_kwargs=dict(layers=[64])
    )
    model_TD.learn(total_timesteps=max_timesteps, log_interval=10)
    model_TD.save(f'{path_TD}/{mname_TD}_{i}')
# testing()
