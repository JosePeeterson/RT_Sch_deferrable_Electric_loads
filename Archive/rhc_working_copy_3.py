from pyomo.environ import *
import numpy as np
import time
import itertools
import sys

m = ConcreteModel()

# constants
a1 = 1
a2 = 0
del_t = 5
M = 2 # set of active tasks
N = 4 # 4 time steps
# note; w_hat is energy over del_t (5 seconds) NOT power
w_hat = np.array([5,5,5,5]).T
E = np.array([1.6,1.6]).T



ones_G = np.ones((1,M))
ones_W = np.ones((M,1))
ones_WG = np.ones((N,1))
maxP = 0.14
d = np.array([20,20]) # deadline in seconds
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

p={}
p[1,1] = 1
p[1,2] = 1
p[1,3] = 1
p[1,4] = 1
p[2,1] = 1
p[2,2] = 1
p[2,3] = 1
p[2,4] = 1
m.phi = Var(m.I, m.K, domain=Reals)
#print(m.phi[2,4])

#m.energy = Param(m.I, m.K,within=Reals)


e={}
e[1,1] = 1
e[1,2] = 1
e[1,3] = 1
e[1,4] = 1
e[2,1] = 1
e[2,2] = 1
e[2,3] = 1
e[2,4] = 1
m.energy = Var(m.I, m.K, domain=NonNegativeReals)


        
m.W = Var(m.I,m.K,domain=NonNegativeReals)
m.G = Var(m.I,m.K,domain=NonNegativeReals)

m.G[1,1] = m.G[1,2]= m.G[1,3] = m.G[1,4] = m.G[2,1] = m.G[2,2] = m.G[2,3] = m.G[2,4] = 0
m.W[1,1] = m.W[1,2] = m.W[1,3] = m.W[1,4] = m.W[2,1] = m.W[2,2] = m.W[2,3] = m.W[2,4] = 0

G_var = np.array([[m.G[1,1], m.G[1,2],m.G[1,3], m.G[1,4] ],[m.G[2,1], m.G[2,2], m.G[2,3], m.G[2,4]]])
W_var = np.array([[m.W[1,1], m.W[1,2], m.W[1,3], m.W[1,4]],[m.W[2,1], m.W[2,2], m.W[2,3], m.W[2,4]]])
#print(np.matmul(ones_G,G_var))

iter_At = range(M)
iter_N = range(N)

## OBJECTIVE function START ##
obj_term1 = np.sum(np.matmul(ones_G,G_var))
obj_term3 = sum( (N - m.phi[i+1,k+1] )**2 for i,k in itertools.product(iter_At,iter_N) )
obj_exp = obj_term1 + obj_term3
m.obj = Objective(expr= obj_exp)
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
        if curr_time + k*del_t <= d[i] + curr_time:
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
            c4_exp = E[i] - sum( m.W[i+1,j+1] + m.G[i+1,j+1] for j in range(k)) - m.energy[i+1,k+1] == 0
            print(c4_exp)
            m.cons.add(expr= c4_exp)
            #print(m.c4[i])
## Fourth constraint END ##


## Fifth constraint START ##
m.c5 = []
for i in range(M):
    for k in range(N):
            c5_exp = m.phi[i+1,k+1] == d[i] - (curr_time + (k*del_t)) - m.energy[i+1,k+1]/maxP
            m.cons.add(expr= c5_exp)
            #print(m.c5[i])
## Fifth constraint END ##



## other way to define constraints
#model.cons1 = ConstraintList()
#model.cons1.add(expr = expression)
#model.cons1[i].pprint()

#m.pprint()

results = SolverFactory("octeract-engine").solve(m,tee=True,keepfiles=False)

results.write()
m.solutions.load_from(results)

for v in m.component_objects(Var, active=True):
    print ("Variable",v)
    varobject = getattr(m, str(v))
    for index in varobject:
        print ("   ",index, varobject[index].value)

