import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

paths = ['grav1.0','grav2.0','grav3.0','grav4.0','T4wT1']
names = ['g=0.0025','g=0.0025*2.0', 'g=0.0025*3.0', 'g=0.0025*4.0', 's2: g1->g4', 's3: g1->g4']
# paths = ['grav1.0']

paths = ['grav1.0','grav4.0','T4wT1']
names = ['T0','T1', 'T0->T1']

window = 50

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

for k,p in enumerate(paths): 
    data = []   
    minl = 5000
    for i in range(1,11):
        data.append(np.load(f'output2/{p}/record_stepsEpisode_{i}.npy'))
        if(len(data[i-1])<minl):
            minl = len(data[i-1])
    ep_steps = np.empty((minl, 0))        
    for i in range(0,10):      
        steps = smooth(data[i][0:minl], window,'flat')     
        steps = steps[0:minl]
        d = steps.reshape(-1,1)
        ep_steps = np.append(ep_steps, d, axis=1)
    print(ep_steps.shape)
    mu = ep_steps.mean(axis=1)
    sigma = ep_steps.std(axis=1) 
    ax1=plt.subplot(1, 1, 1)   
    
    if(minl>375):
        time = np.arange(0,375,1)
        ax1.plot(time, mu[0:375], label=names[k])
        ax1.fill_between(time, mu[0:375]+sigma[0:375], mu[0:375]-sigma[0:375], alpha=0.5)
    else:
        time = np.arange(0,minl,1)    
        ax1.plot(time, mu, label=names[k])
        ax1.fill_between(time, mu+sigma, mu-sigma, alpha=0.5)
    # plt.title('Mountain Car')
    plt.grid(True)
    ax1.set_xlabel('Training Episodes')
    ax1.set_ylabel('Average steps')
    ax1.legend(loc='upper right')
plt.show()