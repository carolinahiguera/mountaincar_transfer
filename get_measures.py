import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

paths = ['grav1.0','grav2.0','grav3.0','grav4.0']
# paths = ['grav4.5','grav5.0']
paths = ['T4wT1_3s_2']

for p in paths:    
    for i in range(1,10):
        print(f'Getting measures for {p} trial {i}')
        dpath = f'output/{p}/DQN_{i}'
        dname = os.listdir(dpath)
        dname.sort()
        ea = EventAccumulator(os.path.join(dpath, dname[0])).Reload()
        tags = ['episode_reward','input_info/rewards','loss/loss', 'loss/td_error']
        labels = {'episode_reward':'episode_reward', 
                'input_info/rewards':'input_info_rewards', 
                'loss/loss':'loss',
                'loss/td_error':'td_error'}

        for tag in tags:
            tag_values=[]
            steps=[]
            for event in ea.Scalars(tag):
                tag_values.append(event.value)        
                steps.append(event.step)
            data = np.column_stack((steps,tag_values))
            np.save(f'{dpath}/{labels[tag]}_{i}.npy', data)
    print('done')
