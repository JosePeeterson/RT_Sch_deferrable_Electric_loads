from pyomo.environ import *
import numpy as np
import time
import itertools

m = ConcreteModel()

# constants
a1 = 1
a2 = 1
del_t = 5
M = 2 # set of active tasks
N = 4 # 4 time steps
# note; w_hat is energy over del_t (5 seconds) NOT power
w_hat = np.array([[5,5,5,5]]).T
E = np.array([[1.6,1.6]]).T
ones_G = np.ones((1,M))
ones_W = np.ones((M,1))
ones_WG = np.ones((N,1))
maxP = 0.14
d = np.array([20,40]) # deadline in seconds
start_time = time.time()
curr_time = 0 #start_time - time.time()


# dimensions of matrix, m x n
m.m = Param(within=NonNegativeIntegers)
m.n = Param(within=NonNegativeIntegers)
m.N = Param(within=NonNegativeIntegers)


#m.m = M
#m.n = N

# index set, indexing the matrix or vector
m.I = RangeSet(1,M)
m.K = RangeSet(1,N)
m.phi = Param(m.I, m.K,within=Reals)
m.energy = Param(m.I, m.K,within=Reals)
        
m.W = Var(m.I,m.K,domain=NonNegativeReals)
m.G = Var(m.I,m.K,domain=NonNegativeReals)

iter_At = range(M)
iter_N = range(N)

m.obj = Objective(expr= np.linalg.norm(np.matmul(ones_G,m.G),ord=1) + np.linalg.norm(np.matmul(ones_G,m.G),ord=np.inf) + sum( (N - m.phi[i,k] )**2 for i,k in itertools.product(iter_At,iter_N) ) )

m.c1 = Constraint(expr = np.matmul((m.W).T,ones_W) <= w_hat)
m.c2 = Constraint(expr = np.matmul((m.W + m.G),ones_WG) == E )


def WG_constraint_rule(m, i,maxP,del_t,curr_time):
    for k in m.K:
        if curr_time + k*del_t <= d[i] + curr_time:
            return (m.W[i,k] + m.G[i,k] <= maxP*del_t)
        else:
            return (m.W[i,k] + m.G[i,k] == 0)

m.c3 = Constraint(m.I, rule=WG_constraint_rule)


def energy_constraint_rule(m, i,maxP, del_t,curr_time,E):
    for k in m.K:
         return (m.energy[i,k] == E[i] - sum( m.W[i,j] + m.G[i,j] for j in k)  ) 

m.c4 = Constraint(m.I, rule=energy_constraint_rule)

def phi_constraint_rule(m, i,maxP, del_t,curr_time):
    for k in m.K:
        return (m.phi[i,k] == d[i] - curr_time + (k*del_t) - m.energy[i,k]/maxP )

m.c5 = Constraint(m.I, rule=phi_constraint_rule)




