from pyomo.environ import *
import numpy as np
import time
import itertools
import sys
import pandas as pd


## INPUTS 
df1 = pd.read_csv('task_set.csv')
task_set = df1.to_numpy()
tot_tasks = task_set.shape[0]
rem_E = task_set[:,1] # remanining energy
a = task_set[:,0]#arrivals in seconds
d = a + 20 # deadlines are fixed 


maxP = 0.14
global del_t
del_t = 5 # it takes 5 seconds to solve optimization every time

## CONSTANTS 

global start_time
start_time = time.time()

# get current time since simulation start
def curr_time():
    return round(time.time() - start_time,2)

# read wind power/(actually it is energy (KwS) as power (Kw) data is very 1 sec.)
global wind_data
df = pd.read_csv('wind_power_data.csv',header=None)
wind_data = df.to_numpy().squeeze()


def wind_pred(N):
    SD = 0.008  # fixed random uncertainty 
    w_hat = np.array([])
    w1 = wind_data[int(curr_time())]
    # note: w_hat is energy over del_t (5 seconds) so energy (KWsec) = power*del_t for N time steps
    w_hat = np.append(w_hat,w1*1) # here should be 1 not del_t
    for i in range(N-1):
        # need to randomize - SD*1 below it could be +ve or -ve
        w_hat = np.append(w_hat,(w1*1 - SD*i)) 
        #print(w_hat)
    return w_hat.T


def calc_M_N_final_d(d,rem_E):
    M = 0 # 1 means NO active tasks because we start at 1 for Rangeset NOT 0. 
    active_d = []
    act_tsk_idx = [] # index of the task set in task_set[]
    for i in range(len(d)):
        if( (curr_time() >= a[i] ) and (curr_time() < d[i]) and rem_E[i] > 0.01): 
            M+=1 # number of active tasks
            active_d.append(d[i])
            act_tsk_idx.append(i)
    # here it takes ~5 seconds to optimize so we -5
    if (M != 0):
        N = np.floor((max(active_d) - curr_time() ) / del_t)
        #N = N + 1 # add +1 as N starts from 1
        final_d = max(active_d) 
    else:
        N = 0 # 
        final_d = d[-1] # ensure that tasks come joined / have some overlap
    #print('M, N =',M,N)
    #reduce the number of time steps to allocate energy by multiples of 5 (time taken to solve optimization)
    
    return M,int(N),final_d, act_tsk_idx



# When we go to the next set of active tasks all the variables (energy phi W_old and G_old) have to be reintilaize to 0.

# do untill final deadline i.e. All tasks are completed 

timer=0    
first_time = True
# reinitialize as above.
# unitl final deadline of set of active tasks 
while(curr_time() >= 5*timer and curr_time() < d[-1]):
    st = curr_time() # start time of optimisation at t
    
    #print('curr_tim',curr_time())

    if first_time:
        first_time = False
    else:
        i=0
        print(old_et - old_st)
        old_k = int(np.ceil((old_et - old_st) / del_t) )

        print('old_k',old_k)

        if old_k > old_N:
            old_k = N 
        for old_t_idx in old_act_tsk_idx:
            print('old remE bef',rem_E[old_t_idx] )      
            rem_E[old_t_idx] -= (sum(old_W[i,:old_k]) + sum(old_G[i,:old_k]))
            print('old sum W',sum(old_W[i,:old_k]))
            i+=1
            print('new remE aft',rem_E[old_t_idx] )        

    M,N,final_d,act_tsk_idx = calc_M_N_final_d(d,rem_E)

    if (M == 0):
        first_time = True
        continue
    print('new N',N)
    print('new M',M)
    w_hat = wind_pred(N)

    ones_G = np.ones((1,M))
    ones_W = np.ones((M,1))
    ones_WG = np.ones((N,1))

    m = ConcreteModel()

    # index set, indexing the matrix or vector
    m.I = RangeSet(1,M)
    m.K = RangeSet(1,N)

    m.W = Var(m.I,m.K,domain=NonNegativeReals)
    m.G = Var(m.I,m.K,domain=NonNegativeReals)

    iter_At = range(M)
    iter_N = range(N)

    G_var = []
    W_var = []
    for i,k in itertools.product(iter_At,iter_N):
        m.G[i+1,k+1] = 0.0
        m.W[i+1,k+1] = 0.0
        G_var.append(m.G[i+1,k+1])
        W_var.append(m.W[i+1,k+1])

    G_var = np.array(G_var)
    W_var = np.array(W_var)
    G_var = np.reshape(G_var,(M,N))
    W_var = np.reshape(W_var,(M,N))



    

    m.phi = Var(RangeSet(1,M), RangeSet(1,N), domain=Reals)
    #print(m.phi[2,4])

    m.energy = Var(RangeSet(1,M), RangeSet(1,N), domain=NonNegativeReals)

    #G_var = np.array([[m.G[1,1], m.G[1,2],m.G[1,3], m.G[1,4] ],[m.G[2,1], m.G[2,2], m.G[2,3], m.G[2,4]]])
    #W_var = np.array([[m.W[1,1], m.W[1,2], m.W[1,3], m.W[1,4]],[m.W[2,1], m.W[2,2], m.W[2,3], m.W[2,4]]])
    #print(np.matmul(ones_G,G_var))

    #iter_At = range(M)
    #iter_N = range(N)

    ## OBJECTIVE function START ##
    obj_term1 = np.sum(np.matmul(ones_G,G_var))
    obj_term3 = sum( (N - m.phi[i+1,k+1] )**2 for i,k in itertools.product(iter_At,iter_N) )
    obj_exp = obj_term1 + obj_term3
    m.obj = Objective(expr= obj_exp,sense= minimize)
    ## OBJECTIVE function END ##

    m.cons = ConstraintList()
    ## First constraint START ##
    m.c1 = []
    for k in range(N):
        c1_exp = np.matmul((W_var).T,ones_W)[k][0] <= w_hat[k]
        m.cons.add(expr= c1_exp)
        #m.c1[k].pprint()
    ## First constraint END ##


    ## second constraint START ##
    m.c2 = []
    for i in range(M):
        c2_exp = np.matmul((W_var + G_var),ones_WG)[i][0] == rem_E[act_tsk_idx[i]]
        #print(np.matmul((W_var + G_var),ones_WG)[i][0])
        m.cons.add(expr= c2_exp)
        #print(m.c2[i])
    ## second constraint END ##


    ## Third constraint START ##
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
    ## Third constraint END ##


    ## Fourth constraint START ##
    m.c4 = []
    for i in range(M):
        for k in range(N):
            c4_exp = rem_E[act_tsk_idx[i]] - sum( m.W[i+1,j+1] + m.G[i+1,j+1] for j in range(k)) - m.energy[i+1,k+1] == 0
            #print(c4_exp)
            m.cons.add(expr= c4_exp)
                #print(m.c4[i])
    ## Fourth constraint END ##


    ## Fifth constraint START ##
    m.c5 = []
    for i in range(M):
        for k in range(N):
            c5_exp = m.phi[i+1,k+1] == d[i] - (curr_time() + (k*del_t)) - m.energy[i+1,k+1]/maxP
            m.cons.add(expr= c5_exp)
                #print(m.c5[i])
    ## Fifth constraint END ##



    ## other way to define constraints
    #model.cons1 = ConstraintList()
    #model.cons1.add(expr = expression)
    #model.cons1[i].pprint()

    #m.pprint()

    results = SolverFactory("octeract-engine").solve(m,tee=True,keepfiles=False)
    et = curr_time() # end time of optimisation at t
    print('Total exec time =', curr_time() + et - st)
    #results.write()
    #m.solutions.load_from(results)
    old_st = st
    old_et = et
    old_N = N
    old_W = np.zeros((M,N))
    old_G = np.zeros((M,N))
    timer+=1

    for i,k in itertools.product(iter_At,iter_N):
        old_W[i,k] = m.W[i+1,k+1].value
        old_G[i,k] = m.G[i+1,k+1].value


    old_act_tsk_idx = act_tsk_idx

    # for v in m.component_objects(Var, active=True):
    #     print ("Variable",v)
    #     varobject = getattr(m, str(v))
    #     for index in varobject:
    #         print ("   ",index, varobject[index].value)
    #         index

            
        


