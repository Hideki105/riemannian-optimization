import numpy as np
import os
import datetime
import time
import copy
from numba import jit
from torch import dtype

class alg441():
  def __init__(self,fn = None,itemax=3000,init="np.svd",c1=1e-10,eps=1e-100,verbose=False):
    if fn == None:
      time = datetime.datetime.now()
      self.fn = time.strftime('%Y%m%d%H%M%S')+".csv"
    else:
      self.fn = fn
    self.init = init
    self.c1   = c1
    self.eps  = eps
    self.itemax= itemax
    self.verbose = verbose
  
  @staticmethod
  @jit(cache=True,forceobj=True)
  def initial_point(A,rank,param):
    m = A.shape[0]
    n = A.shape[1]
    p = rank
    
    if param["init"] == "ones":
      U0 = np.ones((m,p))
      V0 = np.ones((n,p))
      U0 = alg441.multiqr(U0)
      V0 = alg441.multiqr(V0)
    elif param["init"] == "np.svd":
      U0,_, V0T = np.linalg.svd(A, full_matrices=True)
      U0 = U0[:,:p]
      V0 = V0T.T[:,:p]
      U0 = alg441.multiqr(U0)
      V0 = alg441.multiqr(V0)
    elif param["init"]=="np.random":
      np.random.seed(1)
      U0 = np.random.normal(size=(m,p))
      V0 = np.random.normal(size=(n,p))
      U0 = alg441.multiqr(U0)
      V0 = alg441.multiqr(V0)
    else:
      pass
    U0.astype(np.longdouble)
    V0.astype(np.longdouble)
    return U0,V0
  
  @staticmethod
  @jit(cache=True,forceobj=True)
  def multiqr(A):
    #Vectorized QR decomposition.
    q, r = np.vectorize(np.linalg.qr, signature="(m,n)->(m,k),(k,n)")(A)

    # Compute signs or unit-modulus phase of entries of diagonal of r.
    diagonal = np.diagonal(r, axis1=-2, axis2=-1).copy()
    diagonal[diagonal == 0] = 1
    s = diagonal / np.abs(diagonal)

    if A.ndim == 3:
        s = np.expand_dims(s, axis=1)
    q = q * s
    return q

  @staticmethod
  @jit(cache=True,forceobj=True)
  def gradF(A,N,Uk,Vk):
    xik  = A@Vk@N   - Uk@alg441.sym(Uk.T@A@Vk@N)
    etak = A.T@Uk@N - Vk@alg441.sym(Vk.T@A.T@Uk@N)
    return xik,etak

  @staticmethod
  @jit(cache=True,forceobj=True)
  def norm_gradF(A,N,Uk,Vk):
    xik,etak = alg441.gradF(A,N,Uk,Vk)
    J1 = np.linalg.norm(xik)
    J2 = np.linalg.norm(etak)
    return J1+J2

  @staticmethod
  @jit(cache=True,forceobj=True)
  def sym(X):
    return (X+X.T)/2

  @staticmethod
  @jit(cache=True,forceobj=True)
  def R(Uk,Vk,etak,xik,alphak=1e-3):
    Ukp1 = alg441.multiqr(Uk+alphak*xik)
    Vkp1 = alg441.multiqr(Vk+alphak*etak)
    return Ukp1,Vkp1
  
  @staticmethod
  @jit(cache=True,forceobj=True)
  def armijo_rule(A,N,Uk,Vk,etak,xik,Ukp1,Vkp1,alphak,param):
    left = -np.trace(Uk.T@A@Vk@N)\
           +np.trace(Ukp1.T@A@Vkp1@N)\
    
    right = -param["c1"]*(np.trace(etak.T@(alphak*etak))\
                      +np.trace(xik.T@(alphak*xik)))
    return left>=right
  
  @staticmethod
  def savecsv(d_,logpath):
    with open(logpath,"a") as f:
      np.savetxt(f,d_.reshape(1,d_.size), delimiter=",")
    time.sleep(1e-5)
    return

  @staticmethod
  def log(e,n,a,logpath,param):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logpath = os.path.join(BASE_DIR,logpath,param["fn"])
    d_ = np.hstack([e,n,a])
    alg441.savecsv(d_,logpath)
    return
  
  @staticmethod
  @jit(forceobj=True)
  def run(A,N,Uk,Vk,param):
    #step2
    c = 0
    while True:
      c = c+1
      if param["verbose"]:
        if c ==1:
          alphak = np.nan
        alg441.log(-np.trace(Uk.T@A@Vk@N),alg441.norm_gradF(A,N,Uk,Vk),alphak,"log",param)
        print(c,-np.trace(Uk.T@A@Vk@N),alg441.norm_gradF(A,N,Uk,Vk),alphak)
      if c > param["itemax"]:
        break
      xik,etak = alg441.gradF(A,N,Uk,Vk)
      #step4,5
      flg = False
      for alphak in np.logspace(-15,0,16,base=10)[::-1]: 
        if alphak==1:
          alphak = alphak-1e-15
        Ukp1,Vkp1 = alg441.R(Uk,Vk,etak,xik,alphak=alphak)
        if alg441.armijo_rule(A,N,Uk,Vk,etak,xik,Ukp1,Vkp1,alphak,param):
          flg = True
          break
      
      if flg == False:
        alphak=0
        Ukp1,Vkp1 = alg441.R(Uk,Vk,etak,xik,alphak=alphak)
        flg =True
      if alg441.norm_gradF(A,N,Uk,Vk)<param["eps"]:
        break
      #step6
      Uk   = Ukp1
      Vk   = Vkp1
      
    return Uk,Vk

  def set_parameter(self):
    param = {}
    param["fn"] = self.fn
    param["init"] = self.init
    param["c1"]   = self.c1
    param["eps"]  = self.eps
    param["itemax"]= self.itemax
    param["verbose"] = self.verbose
    return param

  def solve(self,A,rank):
    param = self.set_parameter()
    #step1
    Uk,Vk=self.initial_point(A,rank,param)
    
    #l = [i+1 for i in range(rank)]
    #l.reverse()
    l = [1/(i+1) for i in range(rank)]
    N = np.diag(l)
    N.astype(np.longdouble)

    #step 2-6
    Uk,Vk = alg441.run(A,N,Uk,Vk,param)
    return Uk,Vk