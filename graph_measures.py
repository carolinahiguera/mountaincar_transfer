import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

light_orange = (248/255,200/255,173/255)
dark_orange = (249/255,95/255,7/255)

simulations = ['grav1.0','grav2.0','grav3.0','grav4.0','T4wT1', 'T4wT1_3s_2']
names = ['g=0.0025','g=0.0025*2.0', 'g=0.0025*3.0', 'g=0.0025*4.0', 's2: g1->g4', 's3: g1->g4']

simulations = ['grav1.0','grav4.0','T4wT1']
names = ['T0','T1', 'T0->T1']

def smooth(x,window_len=11,window='hanning'):    
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def plot_measures(only_reward=True):
    window = 50
    ini_plot=0
    for r, sim in enumerate(simulations): 
        print(sim)              
        # read from files
        
        timesteps = []
        rewards = []
        lens = []
        # time = list(timesteps)
        # rew = reward[0:len(time)]     
        for i in range(1,10):
            print(i)
            episode_reward = np.load(f'output/{sim}/DQN_{i}/episode_reward_{i}.npy') 
            timesteps.append( episode_reward[:,0] )
            rewards.append( episode_reward[:,1] ) 
            lens.append(rewards[-1].shape[0])           
        idx = lens.index(min(lens))
        time = list(timesteps[idx])
        data = np.empty((lens[idx], 0))        
        for i in range(1,10):
            rs = smooth(rewards[i-1][0:lens[idx]], window,'flat')
            d = rs.reshape(-1,1)
            data = np.append(data, d[0:lens[idx]], axis=1)
        print(data.shape)
        mu = data.mean(axis=1)
        sigma = data.std(axis=1) 
        # rew = list(rew)
        # y_smooth = smooth(mu, window,'flat')
        # reward_smooth = list(y_smooth[0:len(time)])                  
        if only_reward:     
            ax1=plt.subplot(1, 1, 1)    
        else:    
            ax1=plt.subplot(3, 1, 1) 
        # ax1.plot(time, reward_smooth, label=names[r])
        # ax1.fill_between(time, reward_smooth+sigma, reward_smooth-sigma, alpha=0.5)   
        ax1.plot(time, mu, label=names[r])
        ax1.fill_between(time, mu+sigma, mu-sigma, alpha=0.5)
        # plt.title('Mountain Car')
        plt.grid(True)
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Episode reward')
        ax1.legend(loc='lower right')

        if not only_reward:
            # smoothed loss
            idx=0
            floss = np.load(f'output/gravity/{sim}/loss.npy')        
            timesteps = floss[:,0]        
            loss1 = floss[:,1]               
            time = list(timesteps) 
            loss = list(loss1)
            y_smooth = smooth(np.array(loss),window,'flat')
            loss_smooth = list(y_smooth[0:len(time)])
            ax2=plt.subplot(3, 1, 2)        
            ax2.plot(time[ini_plot:], loss_smooth[ini_plot:], label=names[r])
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Loss')
            ax2.legend()

            # smoothed td error
            idx = 0
            f_tderror = np.load(f'output/gravity/{sim}/td_error.npy')                
            timesteps = f_tderror[:,0]        
            td1 = f_tderror[:,1]        
            time = list(timesteps)
            td = list(td1)
            y_smooth = smooth(np.array(td),window,'flat')
            td_smooth = list(y_smooth[0:len(time)])
            ax3=plt.subplot(3, 1, 3)        
            ax3.plot(time[ini_plot:], td_smooth[ini_plot:], label=names[r])
            ax3.set_xlabel('Time')
            ax3.set_ylabel('TD error')
            ax3.legend()    
    plt.show()

def plot_value_s0():
    window = 2000
    ini_plot = 800
    for r in gammas:
        print(f'gamma={r}')
        i = lbl_gammas[r]  
        fvalue1 = np.load(f'./discount_config3/sim{i}/values_s0_iter0.npy')     
        value = fvalue1  
        y_smooth = smooth(np.array(value),window,'flat')
        ax1=plt.gca()
        ax1.plot(y_smooth[ini_plot:], label=r)
        ax1.legend()
    plt.show()


plot_measures(only_reward=True)
# plot_value_s0()
