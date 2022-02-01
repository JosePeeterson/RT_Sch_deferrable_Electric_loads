import numpy as np
import time
import itertools
import sys
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('task_set.csv')
task_set = df1.to_numpy()
rem_E = task_set[:,1]
a = task_set[:,0]#arrivals in seconds
d = a + 20 # deadlines are fixed 
grid_E = np.zeros(task_set.shape[0]) # energy sourced from grid.
# read wind power/(actually it is energy (KwS) as power (Kw) data is very 1 sec.)
global wind_data
df = pd.read_csv('wind_power_data.csv',header=None)
wind_data = df.to_numpy().squeeze()

global start_time
start_time = time.time()

def curr_time():
    return round(time.time() - start_time,2)


while(curr_time() < d[-1]):
    w1 = wind_data[int(curr_time())]

    active_d = []
    act_tsk_idx = [] # index of the task set in task_set[]
    for i in range(len(d)):
            if( (curr_time() >= a[i] ) and (curr_time() < d[i])): 
                active_d.append(d[i])
                act_tsk_idx.append(i)

    earliest_d = np.min(d)
    earliest_d_idx = np.argmin(d)
    sorted_d_idx = np.argsort(d)

    first=True
    remainder_w = 0
    for i in sorted_d_idx:
        if first==True:
            rem_E[i] = rem_E[i] - w1 
            if (rem_E[i] > 0 ): # if there is some residual energy requirement
                grid_E[i] = rem_E[i]
            elif(rem_E[i] < 0):
                grid_E[i] = 0
                remainder_w = w1 - rem_E[i]
        else:
            if(remainder_w > 0):
                rem_E[i] = rem_E[i] - remainder_w 
                grid_E[i] = rem_E[i]
            else:
                grid_E[i] = rem_E[i]

plt.plot(grid_E)
plt.plot(rem_E)
plt.show()
