import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append('./deepq')
sys.path.append('./utils')

import pickle
import gym
import numpy as np
from mountain_car_gravity import MountainCarEnv

from deepq.policies import MlpPolicy, LnMlpPolicy
from deepq.dqn import DQN


with_render = False
path_logs = f'./output/grav4.0/'
model_name = 'mc_model'

env = MountainCarEnv(grav=4.0)
state = env.reset()

num_episodes = 1000
seq_states_episodes = []

model = DQN.load(f'{path_logs}/{model_name}',env)

for i in range(num_episodes):
    print(f'Episode {i}')    
    done = False
    state = env.reset()
    sequence_states = [state]
    while not done:
        action, _ = model.predict(state)
        state, reward, done, _ = env.step(action)
        sequence_states.append(state)
    seq_states_episodes.append(sequence_states)

with open(f'{path_logs}/seq_states_100e.pkl', 'wb') as f:
    pickle.dump(seq_states_episodes, f)