from pymanopt.manifolds import Stiefel, Product
from pymanopt.manifolds.manifold import Manifold
from pymanopt import Problem
import autograd.numpy as np
import pymanopt.manifolds
from manopt.algorithms import ConjugateGradient

def svd_hybrid(A,p,beta_type,init=None):
    m = A.shape[0]
    n = A.shape[1]
    l = [1/(i+1) for i in range(p)]
    N = np.diag(l)
    
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
        beta_type=beta_type,
        verbosity=2,
        log_verbosity=1,
        min_gradient_norm=1e-1000,
        min_step_size=1e-1000,
        max_iterations=3000)

    # run
    if init=="np.svd":
        # initial guess
        U0,_,V0T = np.linalg.svd(A,full_matrices=False) 
        U0, _ = np.linalg.qr(U0[:,:p])
        V0, _ = np.linalg.qr(V0T.T[:,:p])
        x0 = np.array([U0, V0])

        res = solver.run(problem, initial_point=x0)
        U,V = res.point
    else:
        res = solver.run(problem)
        U,V = res.point
    return U,V,res