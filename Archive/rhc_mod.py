from pyomo.environ import *
import numpy as np
import time
import itertools
import sys

m = ConcreteModel()

M = 2 # set of active tasks
N = 4 # 4 time steps

ones_G = np.ones((1,M))
ones_W = np.ones((M,1))

#m.m = Param(within=NonNegativeIntegers)
#m.n = Param(within=NonNegativeIntegers)

m.I = RangeSet(1,M)
m.K = RangeSet(1,N)


v={}
v[1,1] = 1
v[1,2] = 1
v[1,3] = 1
v[1,4] = 1
v[2,1] = 1
v[2,2] = 1
v[2,3] = 1
v[2,4] = 1
m.phi = Param(m.I, m.K, initialize=v, default=0)
print(m.phi[1,1])

m.W = Var(m.I,m.K,domain=NonNegativeReals)
m.G = Var(m.I,m.K,domain=NonNegativeReals)


G_var = np.array([[m.G[1,1], m.G[1,2]],[m.G[2,1], m.G[2,2]]])
W_var = np.array([[m.W[1,1], m.W[1,2]],[m.W[2,1], m.W[2,2]]])

iter_At = range(M)
iter_N = range(N)

#print(np.linalg.norm(np.matmul(ones_G,G_var),ord=1))

obj_exp =  abs(m.G[1,1] + m.G[2,1])



m.x1 = Var(within=Reals,bounds=(0,1),initialize=1)
m.x2 = Var(within=Reals,bounds=(0,1),initialize=1)
m.x3 = Var(within=Reals,bounds=(0,1),initialize=None)

x1 = m.x1
x2 = m.x2
x3 = m.x3

m.obj = Objective(expr=obj_exp)

m.e3 = Constraint(expr = 20*x1 + 12*x2 + 11*x3 <= 40)
sys.exit()
