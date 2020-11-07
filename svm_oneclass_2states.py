import pickle
import numpy as np
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt 

path_task1 = f'./output/grav1.0/'
path_task2 = f'./output/grav2.0/'
path_task3 = f'./output/grav3.0/'
path_task4 = f'./output/grav4.0/'

with open(f'{path_task1}/seq_states_100e.pkl', 'rb') as f:
    seq_states = pickle.load(f)

with open(f'{path_task2}/seq_states_100e.pkl', 'rb') as f:
    seq_states_t2 = pickle.load(f)

with open(f'{path_task3}/seq_states_100e.pkl', 'rb') as f:
    seq_states_t3 = pickle.load(f)

with open(f'{path_task4}/seq_states_100e.pkl', 'rb') as f:
    seq_states_t4 = pickle.load(f)

dataset_task1 = np.empty((0,4))
dataset_task2 = np.empty((0,4))
dataset_task3 = np.empty((0,4))
dataset_task4 = np.empty((0,4))

def get_datasets():
    global dataset_task1, dataset_task2, dataset_task3, dataset_task4
    for i in range( len(seq_states)):
        for j in range(1,len(seq_states[i])):
            seq = [seq_states[i][j-1][0], seq_states[i][j-1][1], seq_states[i][j][0], seq_states[i][j][1]]
            dataset_task1 = np.append(dataset_task1, np.array([seq]), axis=0)
    for i in range( len(seq_states_t2)):
        for j in range(1,len(seq_states_t2[i])):
            seq = [seq_states_t2[i][j-1][0], seq_states_t2[i][j-1][1], seq_states_t2[i][j][0], seq_states_t2[i][j][1]]
            dataset_task2 = np.append(dataset_task2, np.array([seq]), axis=0)
    for i in range( len(seq_states_t3)):
        for j in range(1,len(seq_states_t3[i])):
            seq = [seq_states_t3[i][j-1][0], seq_states_t3[i][j-1][1], seq_states_t3[i][j][0], seq_states_t3[i][j][1]]
            dataset_task3 = np.append(dataset_task3, np.array([seq]), axis=0)
    for i in range( len(seq_states_t4)):
        for j in range(1,len(seq_states_t4[i])):
            seq = [seq_states_t4[i][j-1][0], seq_states_t4[i][j-1][1], seq_states_t4[i][j][0], seq_states_t4[i][j][1]]
            dataset_task4 = np.append(dataset_task4, np.array([seq]), axis=0)

def compute_confusion_matrix(clfs):
    conf_matrix = np.zeros((4,4))
    for t in range(4):
        for j in range(4):
            y_pred = clfs[t].predict(Xtest[j])
            if(j==t):
                n_error = y_pred[y_pred == -1].size / Xtest[j].shape[0]
            else:
                n_error = y_pred[y_pred == 1].size / Xtest[j].shape[0]
            conf_matrix[t][j] = n_error
    return conf_matrix

def compute_small_confusion_matrix(clfs):
    tasks = [0,3]
    conf_matrix = np.zeros((len(tasks),len(tasks)))
    for t,task_0 in enumerate(tasks):
        for j,task_1 in enumerate(tasks):
            y_pred = clfs[t].predict(Xtest[task_1])
            if(j==t):
                n_error = y_pred[y_pred == -1].size / Xtest[task_1].shape[0]
            else:
                n_error = y_pred[y_pred == 1].size / Xtest[task_1].shape[0]
            conf_matrix[t][j] = n_error
    return conf_matrix


get_datasets()


# nu = [10e-4, 10e-3, 10e-2, 10e-1, 0.2, 0.5, 0.7, 0.9]
# kernel = ['linear', 'sigmoid']
# degree = [2,3,4,5]
# gamma = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e-0]

nu = [0.001]
kernel = ['rbf',]
degree = [2,3,4,5]
gamma = [100]

# param_nu = 0.002#0.1
# param_kernel = 'rbf'#'rbf'
# param_degree = 2
# param_gamma = 10#0.005

Xtrain = []
Xtest = []
clfs = []

Xtrain.append(dataset_task1[0:int(0.7*dataset_task1.shape[0])])
Xtest.append(dataset_task1[int(0.7*dataset_task1.shape[0])+1:dataset_task1.shape[0]])
Xtrain.append(dataset_task2[0:int(0.7*dataset_task2.shape[0])])
Xtest.append(dataset_task2[int(0.7*dataset_task2.shape[0])+1:dataset_task2.shape[0]])
Xtrain.append(dataset_task3[0:int(0.7*dataset_task3.shape[0])])
Xtest.append(dataset_task3[int(0.7*dataset_task3.shape[0])+1:dataset_task3.shape[0]])
Xtrain.append(dataset_task4[0:int(0.7*dataset_task4.shape[0])])
Xtest.append(dataset_task4[int(0.7*dataset_task4.shape[0])+1:dataset_task4.shape[0]])

path = f'./output/one_svm/'

tasks = [0,3]

for param_kernel in kernel:
    save_path = f'{path}/{param_kernel}/'
    for param_nu in nu:
        for param_gamma in gamma:            
            if(param_kernel!='poly'):
                print(f'kernel={param_kernel} - gamma={param_gamma} - nu={param_nu}')
                clfs = []
                for i,task in enumerate(tasks):
                    clfs.append(svm.OneClassSVM(nu=param_nu, kernel=param_kernel, gamma=param_gamma))
                    clfs[i].fit(Xtrain[task])
                    # pkl_filename = f'svm_model_T{i}.pkl'
                    # with open(pkl_filename, 'wb') as file:
                    #     pickle.dump(clfs[i], file)
                conf_matrix = compute_small_confusion_matrix(clfs)
                fig = plt.gcf()
                ax = plt.subplot()
                sns.heatmap(conf_matrix, annot=True, ax = ax, cmap="YlGnBu");
                ax.xaxis.set_ticklabels(['T0', 'T1', 'T3', 'T4']); ax.yaxis.set_ticklabels(['T0', 'T1', 'T3', 'T4']);
                plt.savefig(f'{save_path}/T0T1_n{param_nu}_r{param_gamma}.png')
                plt.clf()                
            else:
                for param_degree in degree:
                    print(f'kernel={param_kernel} - gamma={param_gamma} - nu={param_nu} - d={param_degree}')
                    clfs = []
                    for i in range(4):
                        clfs.append(svm.OneClassSVM(nu=param_nu, kernel=param_kernel, gamma=param_gamma, degree=param_degree))
                        clfs[i].fit(Xtrain[i])
                    conf_matrix = compute_confusion_matrix(clfs)
                    fig = plt.gcf()
                    ax = plt.subplot()
                    sns.heatmap(conf_matrix, annot=True, ax = ax, cmap="YlGnBu");
                    ax.xaxis.set_ticklabels(['T1', 'T2', 'T3', 'T4']); ax.yaxis.set_ticklabels(['T1', 'T2', 'T3', 'T4']);
                    plt.savefig(f'{save_path}/n{param_nu}_r{param_gamma}_d{param_degree}.png')
                    plt.clf()




# clfs.append(svm.OneClassSVM(nu=param_nu, kernel=param_kernel, gamma=param_gamma, degree=param_degree))
# clfs[0].fit(Xtrain[0])


# clfs.append(svm.OneClassSVM(nu=param_nu, kernel=param_kernel, gamma=param_gamma))
# clfs[1].fit(Xtrain[1])

# )
# clfs.append(svm.OneClassSVM(nu=param_nu, kernel=param_kernel, gamma=param_gamma))
# clfs[2].fit(Xtrain[2])


# clfs.append(svm.OneClassSVM(nu=param_nu, kernel=param_kernel, gamma=param_gamma))
# clfs[3].fit(Xtrain[3])

# conf_matrix = np.zeros((4,4))
# for t in range(4):
#     for j in range(4):
#         y_pred = clfs[t].predict(Xtest[j])
#         if(j==t):
#             n_error = y_pred[y_pred == -1].size / Xtest[j].shape[0]
#         else:
#             n_error = y_pred[y_pred == 1].size / Xtest[j].shape[0]
#         conf_matrix[t][j] = n_error

# ax = plt.subplot()
# sns.heatmap(conf_matrix, annot=True, ax = ax, cmap="YlGnBu");
# ax.xaxis.set_ticklabels(['T1', 'T2', 'T3', 'T4']); ax.yaxis.set_ticklabels(['T1', 'T2', 'T3', 'T4']);
# plt.savefig('conf.png')
