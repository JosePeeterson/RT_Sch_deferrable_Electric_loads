from pyomo.environ import *
import numpy as np
import time
import itertools
import sys
import pandas as pd


## INPUTS and ## CONSTANTS 
df1 = pd.read_csv('task_set.csv') # task set extraction
task_set = df1.to_numpy()
tot_tasks = task_set.shape[0]
rem_E = task_set[:,1] # remanining energy
a = task_set[:,0]#arrivals in seconds
d = a + 20 # deadlines are fixed 
maxP = 0.14 # maximum chargeable/serviceable power

global del_t
del_t = 5 # it takes 5 seconds to solve optimization every time

# record simulation start time
global start_time
start_time = time.time()

# get current time since simulation start
def curr_time():
    return round(time.time() - start_time,2)

# read wind power/(actually it is same as energy (KwS) as power (Kw) data is every 1 sec.)
global wind_data
df = pd.read_csv('wind_power_data.csv',header=None)
wind_data = df.to_numpy().squeeze()


# make predictions of future N-1 time steps
def wind_pred(N):
    SD = 0.008  # Standard Deviation to account for uncertainty 
    w_hat = np.array([])
    # obtain current wind power
    w1 = wind_data[int(curr_time())]
    w_hat = np.append(w_hat,w1*1) # power w1 times 1 sec is energy so in our case power = energy.
    # make wind power predictions.
    for i in range(N-1):
        # multiples of SD increase deviation for future time steps from present wind power
        w_hat = np.append(w_hat,(w1*1 - SD*i)) 
    return w_hat.T

# compute dimensions (M x N) of the matrix W and G 
def calc_M_N_final_d(d,rem_E):
    M = 0 # initialize with 0 tasks
    active_d = [] # list of active tasks deadlines
    act_tsk_idx = [] # index of the task set in task_set[]
    for i in range(len(d)):
        if( (curr_time() >= a[i] ) and (curr_time() < d[i]) and rem_E[i] > 0.01): 
            M+=1 # number of active tasks
            active_d.append(d[i])
            act_tsk_idx.append(i)
    if (M != 0): # for non zero active tasks
        N = np.floor((max(active_d) - curr_time() ) / del_t)
        final_d = max(active_d) 
    else: # for zero active tasks
        N = 0 # 
        final_d = d[-1] # ensure that tasks come joined / have some overlap    
    return M,int(N),final_d, act_tsk_idx


counter=0 # integer to multiply 5 seconds so that optimization only performed after multiples of 5 seconds
first_time = True # flag to record W and G in the second time of optimization for a new active task set
# Do unitll final deadline of set of active tasks 
while(curr_time() < max(d)):
    while(curr_time() >= 5*counter and curr_time() < max(d)):
        st = curr_time() # start time of this optimisation

        if first_time:
            first_time = False
        else:
            # find out how much current time (in terms of time step) has passed since previous optimization calculation
            # Energy up to this point is considered to have been supplied so rem_E has to be updated in the current optimzation calculation
            old_k = int(np.ceil((old_et - old_st) / del_t) )
            if old_k > old_N: # if the present time exceeds final time of previous optimzation (due to slow optimization calculation)
                old_k = N # it means all energy up to N has been applied
            i=0 # task index of previous W and G
            for old_t_idx in old_act_tsk_idx: # index from previous active task set
                # update remaining energy, rem_E (Capital E in paper [2])    
                rem_E[old_t_idx] -= (sum(old_W[i,:old_k]) + sum(old_G[i,:old_k]))
                i+=1
        
        # calculate M N for updated rem_E
        M,N,final_d,act_tsk_idx = calc_M_N_final_d(d,rem_E)

        if (M == 0): # do not solve if there are no active tasks
            first_time = True # set flag to true
            continue
        else:
            print('M =',M)
            print('N =',N)

        w_hat = wind_pred(N) # wind energies over N time steps

        ones_G = np.ones((1,M)) # ones vector
        ones_W = np.ones((M,1)) # ones vector
        ones_WG = np.ones((N,1)) # ones vector

        m = ConcreteModel() # create model to define optimization problem

        # index set, indexing the matrix or vector
        m.I = RangeSet(1,M)
        m.K = RangeSet(1,N)

        # define entries of matrix W and G as variables
        m.W = Var(m.I,m.K,domain=NonNegativeReals)
        m.G = Var(m.I,m.K,domain=NonNegativeReals)

        # iterable of size M and N
        iter_At = range(M)
        iter_N = range(N)

        # array to encapsulate the variables defined above
        G_var = []
        W_var = []
        # Before every optimization the variables in W and G have to be reinitilaized to 0.
        for i,k in itertools.product(iter_At,iter_N):
            m.G[i+1,k+1] = 0.0
            m.W[i+1,k+1] = 0.0
            G_var.append(m.G[i+1,k+1])
            W_var.append(m.W[i+1,k+1])

        G_var = np.array(G_var)
        W_var = np.array(W_var)
        G_var = np.reshape(G_var,(M,N)) # reshape 1d to 2d array
        W_var = np.reshape(W_var,(M,N)) # reshape 1d to 2d array

        # varianle phi in constraint 10
        m.phi = Var(RangeSet(1,M), RangeSet(1,N), domain=Reals)

        # varianle small e in constraint 11
        m.energy = Var(RangeSet(1,M), RangeSet(1,N), domain=NonNegativeReals)

        ## OBJECTIVE function START ##
        obj_term1 = np.sum(np.matmul(ones_G,G_var))
        obj_term3 = sum( (N - m.phi[i+1,k+1] )**2 for i,k in itertools.product(iter_At,iter_N) )
        obj_exp = obj_term1 + obj_term3 # ignore 2nd term in objective function as max() cannot be defined and alpha1 can be set to 1 and alpha2 to 0 
        m.obj = Objective(expr= obj_exp,sense= minimize)
        ## OBJECTIVE function END ##

        m.cons = ConstraintList()
        ## constraint 7 START ##
        m.c1 = []
        for k in range(N):
            c1_exp = np.matmul((W_var).T,ones_W)[k][0] <= w_hat[k]
            m.cons.add(expr= c1_exp)
            #m.c1[k].pprint()
        ## constraint 7 END ##


        ## constraint 8 START ##
        m.c2 = []
        for i in range(M):
            c2_exp = np.matmul((W_var + G_var),ones_WG)[i][0] == rem_E[act_tsk_idx[i]]
            #print(np.matmul((W_var + G_var),ones_WG)[i][0])
            m.cons.add(expr= c2_exp)
            #print(m.c2[i])
        ## constraint 8 END ##


        ## constraint 9 START ##
        m.c3 = []
        for i in range(M):
            for k in range(N):
                if curr_time() + k*del_t <= d[i] + curr_time():
                    c3_exp = m.W[i+1,k+1] + m.G[i+1,k+1] <= maxP*del_t
                    m.cons.add(expr= c3_exp)
                    #print(m.c3[i])
                else:
                    c3_exp = m.W[i+1,k+1] + m.G[i+1,k+1] == 0
                    m.cons.add(expr= c3_exp)
                    #print(m.c3[i])
        ## constraint 9 END ##


        ## constraint 11 START ##
        m.c4 = []
        for i in range(M):
            for k in range(N):
                c4_exp = rem_E[act_tsk_idx[i]] - sum( m.W[i+1,j+1] + m.G[i+1,j+1] for j in range(k)) - m.energy[i+1,k+1] == 0
                #print(c4_exp)
                m.cons.add(expr= c4_exp)
                    #print(m.c4[i])
        ## constraint 11 END ##


        ## constraint 10 START ##
        m.c5 = []
        for i in range(M):
            for k in range(N):
                c5_exp = m.phi[i+1,k+1] == d[i] - (curr_time() + (k*del_t)) - m.energy[i+1,k+1]/maxP
                m.cons.add(expr= c5_exp)
                    #print(m.c5[i])
        ## constraint 10 END ##

        # invoke solver and 
        results = SolverFactory("octeract-engine").solve(m,tee=True,keepfiles=False)  
        #results.write()
        #m.solutions.load_from(results)
        
        et = curr_time() # end time of this optimisation
        print('optimisation time =', et - st)
        print('current time =', curr_time())
        # save current solution for next optimization to recalcualte rem_E in the next optimization
        old_st = st
        old_et = et
        old_N = N
        old_W = np.zeros((M,N)) # save a copy of current optimzation solution in old_W
        old_G = np.zeros((M,N)) # save a copy of current optimzation solution in old_G
        counter+=1
        old_act_tsk_idx = act_tsk_idx

        # transfer every entry of current solution
        for i,k in itertools.product(iter_At,iter_N):
            old_W[i,k] = m.W[i+1,k+1].value
            old_G[i,k] = m.G[i+1,k+1].value

        # display results in terminal
        # for v in m.component_objects(Var, active=True):
        #     print ("Variable",v)
        #     varobject = getattr(m, str(v))
        #     for index in varobject:
        #         print ("   ",index, varobject[index].value)
        #         index
            
print('current time =', curr_time())      


