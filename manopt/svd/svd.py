from pymanopt.manifolds import Stiefel, Product
from pymanopt.manifolds.manifold import Manifold
from pymanopt import Problem
import autograd.numpy as np
import pymanopt.manifolds
from pymanopt.optimizers import ConjugateGradient

np.random.seed(1)

def svd(A,p,init=None):
    m = A.shape[0]
    n = A.shape[1]
    N = np.diag(np.arange(p, 0, -1))
    
    # setting manifold
    stiefel_1 = Stiefel(m, p)
    stiefel_2 = Stiefel(n, p)
    manifold = Product([stiefel_1, stiefel_2])

    # define cost function
    @pymanopt.function.autograd(manifold)
    def cost(U, V):
        return -np.trace(U.T @ A @ V @ N)

    # define problem and solver
    problem = Problem(manifold=manifold, cost=cost)
    solver = ConjugateGradient(
    verbosity=2,
    log_verbosity=0,
    beta_rule="PolakRibiere",
    )
    # initial guess
    U0,_,V0T = np.linalg.svd(A,full_matrices=False) 
    #U0, _ = np.linalg.qr(U0[:,:p])
    #V0, _ = np.linalg.qr(V0T.T[:,:p])
    V0 = V0T.T
    x0 = np.array([U0, V0])

    # run
    if init=="np.svd":
        U,V = solver.run(problem, initial_point=x0).point
    else:
        U,V = solver.run(problem).point
    return U,V
