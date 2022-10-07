import autograd.numpy as np

import pymanopt
from pymanopt.manifolds import Stiefel
from manopt.algorithms import ConjugateGradient
from sklearn.decomposition import FastICA

def ica_hybrid_stiefel(D,beta_type):
    ica = FastICA(n_components=D.shape[1])
    S_ = ica.fit_transform(D)
    
    def create_cost(matrices):
        @pymanopt.function.autograd(manifold)
        def cost(X):
            _sum = 0.
            for matrix in matrices:
                Y = X.T @ matrix @ X
                _sum += - np.linalg.norm(np.diag(Y)) ** 2
            return _sum
        return cost

    def set_matrices(D):
        matrices = []
        lis = [i for i in range(D.shape[0])]
        
        for k in lis:
            for l in lis:
                Ekl=np.zeros([D.shape[0],D.shape[0]])
                Elk=np.zeros([D.shape[0],D.shape[0]])
                Ekl[k,l]=1
                Elk[l,k]=1

                if k > l:
                    pass
                else:
                    if k==l:
                        Mkl = Ekl
                    if k < l:
                        Mkl = (Ekl+Elk)/np.sqrt(2)
                    
                    tmp = np.zeros([D.shape[0],D.shape[0]])
                    for i in [i for i in range(D.shape[1])]:
                        z = S_[:,i].reshape(D.shape[0],1)
                        tmp += (z.T@Mkl@z)*z@z.T
                    tmp = tmp/D.shape[1]
                    C = tmp - np.trace(Mkl)*np.eye(D.shape[0]) - Mkl - Mkl.T
                    matrices.append(C)
        return matrices
    
    #print(D)
    manifold = Stiefel(D.shape[0], D.shape[1])
    matrices = set_matrices(D)
    cost = create_cost(matrices)
    
    problem = pymanopt.Problem(manifold, cost=cost)
    solver = ConjugateGradient(beta_type=beta_type
                                      ,max_iterations=5
                                      ,verbosity=2
                                      ,log_verbosity=1
                                      ,min_gradient_norm=1e-100
                                      ,min_step_size=1e-100,)

    

    res = solver.run(problem,initial_point=S_)
    return res
            