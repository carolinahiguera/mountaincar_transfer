import sys
sys.path.append('./deepq')
sys.path.append('./utils')

import pickle
import gym
import numpy as np
# from mountain_car import MountainCarEnv
from mountain_car_gravity import MountainCarEnv

from deepq.policies import MlpPolicy, LnMlpPolicy
from deepq.dqn import DQN

with_render = False
path_logs = f'output/T4wT1_3s_2/'
model_name = 'transfer_model'
# path_logs = f'output/grav4.0/'
# model_name = 'mc_model'
max_trials = 100

env = MountainCarEnv(grav=4.0)

d_mean = []
d_std = []

for j in range(1,10):
	rewards = np.zeros(max_trials)
	state = env.reset()
	model = DQN.load(f'{path_logs}/{model_name}_{j}', env)
	for i in range(max_trials):
		print(f'Model {j} Test {i}')
		done = False
		state = env.reset()
		while not done:
			action, _ = model.predict(state)
			state, reward, done, _ = env.step(action)
			rewards[i] += reward
	d_mean.append(np.mean(rewards))
	d_std.append(np.std(rewards))

# phi = 0.1
# task = 4
# source = 1
# path_TS = f'output/grav{source}.0/'
# mname_TS = 'mc_model'

# path_TD = f'output/T{task}wT{source}/'
# mname_TD = 'transfer_model'

# pkl_filename = f'{path_TD}/svm_model_T{source}.pkl'
# with open(pkl_filename, 'rb') as file:
#     svm_TS = pickle.load(file)

# for j in range(1,10):
# 	rewards = np.zeros(max_trials)	
# 	model_TD = DQN.load(f'{path_TD}/{mname_TD}_{j}', env)
# 	model_TS = DQN.load(f'{path_TS}/{mname_TS}_{j}', env)	
# 	for i in range(max_trials):
# 		print(f'Model {j} Test {i}')
# 		done = False
# 		obs = env.reset()
# 		action, _ = model_TD.predict(obs)		
# 		new_obs, reward, done, _ = env.step(action)
# 		sequence = np.array([obs[0], obs[1], new_obs[0], new_obs[1]])
# 		while not done:
# 			if(svm_TS.predict(sequence.reshape(1,-1))[0]==1):
# 				Qk_1 = phi * model_TS.predict(new_obs)[1][0]
# 				Qk = (1.0-phi) * model_TD.predict(new_obs)[1][0]
# 				action = np.argmax(Qk_1 + Qk)
# 			else:
# 				action, _ = model_TD.predict(new_obs)			
# 			new_obs, reward, done, _ = env.step(action)
# 			rewards[i] += reward
# 			sequence[0] = sequence[2]
# 			sequence[1] = sequence[3]
# 			sequence[2] = new_obs[0]
# 			sequence[3] = new_obs[1]
# 	d_mean.append(np.mean(rewards))
# 	d_std.append(np.std(rewards))


print(f'Mean: {d_mean}')
print(f'Std: {d_std}')