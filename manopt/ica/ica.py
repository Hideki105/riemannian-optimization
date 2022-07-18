import pymanopt
from pymanopt.manifolds import Stiefel
import autograd.numpy as np
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

def cost(W):
        tmp = np.dot(W, matirx)
        res = (tmp**4).sum()
        return res

def ica(matirx,n):
    matrix = matirx.reshape(n, -1)
    
    manifold = Stiefel(n, n)
    problem = Problem(manifold = manifold, cost = cost, verbosity = 1)
    solver = SteepestDescent()
    W_opt = solver.solve(problem)
    return W_opt
    
