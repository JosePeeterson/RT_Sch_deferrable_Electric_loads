
from pyomo.environ import *

model = AbstractModel()

# dimensions of matrix, m x n
model.m = Param(within=NonNegativeIntegers)
model.n = Param(within=NonNegativeIntegers)
model.N = Param(within=NonNegativeIntegers)

# index set, indexing the matrix or vector
model.I = RangeSet(1,model.m)
model.J = RangeSet(1,model.n)



# matrix defintion or Known coeffecients 
model.a = Param(model.I,model.J)
# vector defintion or Known coeffecients 
model.b = Param(model.I)
model.c = Param(model.J)

#model.W = Var(model.I,model.J,domain=NonNegativeReals)
#model.G = Var(model.I,model.J,domain=NonNegativeReals)

model.x = Var(model.J,domain=NonNegativeReals)

def obj_expression(m):
    return summation(m.c, m.x)

model.OBJ = Objective(rule=obj_expression)

def ax_constraint_rule(m, i):
    # return the expression for the constraint for i
    return sum(m.a[i,j] * m.x[j] for j in m.J) >= m.b[i]










