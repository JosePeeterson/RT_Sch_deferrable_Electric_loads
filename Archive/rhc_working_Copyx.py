from pyomo.environ import *
import numpy as np
import time
import itertools
import sys
import pandas as pd


## INPUTS 
a = np.array([0,0]) # deadline in seconds
d = a + 20
E = np.array([1.6,1.6]).T
maxP = 0.14
global del_t
del_t = 5 # it takes 5 seconds to solve optimization every time

## CONSTANTS 

global start_time
start_time = time.time()

def curr_time():
    return round(time.time() - start_time,2)

global wind_data
df = pd.read_csv('wind_power_data.csv',header=None)
wind_data = df.to_numpy().squeeze()

def wind_pred(N):
    SD = 0.008  # fixed random uncertainty 
    w_hat = np.array([])
    w1 = wind_data[int(curr_time())]
    # note: w_hat is energy over del_t (5 seconds) so energy (KWsec) = power*del_t for N time steps
    w_hat = np.append(w_hat,w1*del_t)
    for i in range(N-1):
        # need to randomize - SD*1 below it could be +ve or -ve
        w_hat = np.append(w_hat,(w1*del_t - SD*i)) 
        #print(w_hat)
    return w_hat.T


def calc_M_N_final_d(d):
    M = 0 # 1 means NO active tasks because we start at 1 for Rangeset NOT 0. 
    active_d = []
    for i in range(len(d)):
        if( (d[i] - curr_time()) > 0 and (d[i] - curr_time()) <= 20 ): 
            M+=1 # number of active tasks
            active_d.append(d[i])
    # here it takes ~5 seconds to optimize so we -5
    if (M != 0):
        N = np.ceil((max(active_d) - curr_time() ) / del_t)
        #N = N + 1 # add +1 as N starts from 1
        final_d = max(active_d) 
    else:
        N = 0 # 
        final_d = d[-1] # ensure that tasks come joined / have some overlap
    #print('M, N =',M,N)
    #reduce the number of time steps to allocate energy by multiples of 5 (time taken to solve optimization)
    
    return M,int(N),final_d

def curr_k(final_d,old_N):
    # constraints 10 and 11 do NOT refer to ALL k but only k at the present time.
    k = np.ceil((curr_time() - (final_d - old_N*del_t)) / del_t)
    return int(k)

# When we go to the next set of active tasks all the variables (energy phi W_old and G_old) have to be reintilaize to 0.

# do untill All tasks are completed
timer=0
while(curr_time() < d[-1]):
    M,N,Active_task_final_d = calc_M_N_final_d(d)
    #print('N',N)
    #print('M',M)
    
    first_time = True
    # reinitialize as above.
    # unitl final deadline of set of active tasks 
    while(M != 0 and curr_time() >= 5*timer and curr_time() < d[-1]):
        print('m',M)
        print('curr_tim',curr_time())

        M,N,final_d = calc_M_N_final_d(d)
        print('m',M)
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
        if first_time:
            G_var = []
            W_var = []
            old_N = N
            old_M = M
            old_final_d = final_d
            for i,k in itertools.product(iter_At,iter_N):
                m.G[i+1,k+1] = 0.0
                m.W[i+1,k+1] = 0.0
                G_var.append(m.G[i+1,k+1])
                W_var.append(m.W[i+1,k+1])
            first_time = False

        G_var = np.array(G_var)
        W_var = np.array(W_var)
        G_var = np.reshape(G_var,(M,N))
        W_var = np.reshape(W_var,(M,N))



        m.phi = Var(RangeSet(1,old_M), RangeSet(1,old_N), domain=Reals)
        #print(m.phi[2,4])

        m.energy = Var(RangeSet(1,old_M), RangeSet(1,old_N), domain=NonNegativeReals)

        #G_var = np.array([[m.G[1,1], m.G[1,2],m.G[1,3], m.G[1,4] ],[m.G[2,1], m.G[2,2], m.G[2,3], m.G[2,4]]])
        #W_var = np.array([[m.W[1,1], m.W[1,2], m.W[1,3], m.W[1,4]],[m.W[2,1], m.W[2,2], m.W[2,3], m.W[2,4]]])
        #print(np.matmul(ones_G,G_var))

        iter_At = range(M)
        iter_N = range(N)

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
            c2_exp = np.matmul((W_var + G_var),ones_WG)[i][0] == E[i]
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
        for i in range(old_M):
            k = curr_k(old_final_d,old_N)
            print('k',k)
            c4_exp = E[i] - sum( m.W[i+1,j+1] + m.G[i+1,j+1] for j in range(k)) - m.energy[i+1,k+1] == 0
            #print(c4_exp)
            m.cons.add(expr= c4_exp)
                    #print(m.c4[i])
        ## Fourth constraint END ##


        ## Fifth constraint START ##
        m.c5 = []
        for i in range(old_M):
            k = curr_k(old_final_d,old_N)
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

        #results.write()
        #m.solutions.load_from(results)
        print('old_n',old_N)
        print('old_m',old_M)
        old_N = N
        old_M = M
        old_final_d = final_d
        timer+=1

        for v in m.component_objects(Var, active=True):
            print ("Variable",v)
            varobject = getattr(m, str(v))
            for index in varobject:
                print ("   ",index, varobject[index].value)
                index

            
        print('Total exec time =', curr_time())

