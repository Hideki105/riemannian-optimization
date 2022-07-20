import numpy as np
import os
import datetime
import time
from numba import jit

class alg445():
  def __init__(self,fn = None,verbose=False):
    if fn == None:
      time = datetime.datetime.now()
      self.fn = time.strftime('%Y%m%d%H%M%S')+".csv"
    else:
      self.fn = fn
    self.verbose = verbose
  
  @staticmethod
  @jit(cache=True,forceobj=True)
  def _R(uk,vk,etak,xik):
    uk = (uk+xik )/np.linalg.norm(uk+xik )
    vk = (vk+etak)/np.linalg.norm(vk+etak)
    return uk,vk
  
  @staticmethod
  @jit(cache=True,forceobj=True)
  def sym(X):
    return (X+X.T)/2

  @staticmethod
  @jit(cache=True,forceobj=True)
  def gradF(A,N,U0,V0):
    xi0  = A@V0@N   - U0@alg445.sym(U0.T@A@V0@N)
    eta0 = A.T@U0@N - V0@alg445.sym(V0.T@A.T@U0@N)
    return xi0,eta0

  @staticmethod
  @jit(cache=True,forceobj=True)
  def norm_gradF(A,N,U0,V0):
    xi0,eta0 = alg445.gradF(A,N,U0,V0)
    J1 = np.linalg.norm(xi0)
    J2 = np.linalg.norm(eta0)
    return J1+J2

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
    alg445.savecsv(d_,logpath)
    return

  @staticmethod
  @jit(forceobj=True)
  def run(A,uk,vk,Uk,Vk,param):
    m = uk.shape[0]
    n = vk.shape[0]
    Im  = np.eye(m) 
    In  = np.eye(n)
    uk  = uk.reshape(m,1)
    ukT = uk.reshape(1,m)
    vk  = vk.reshape(n,1)
    vkT = vk.reshape(1,n)

    #step2
    for k in range(10):
      #step3
      sk = ukT@A@vk
      #d = sk*In - 1/sk*(In-vk@vkT)@A.T@(Im-uk@ukT)@A
      #e = 1/sk*(In-vk@vkT)@A.T@(Im-uk@ukT)@A@vk + (In-vk@vkT)@A.T@uk
      d = sk**2*In-(In-vk@vkT)@A.T@(Im-uk@ukT)@A
      e = (In-vk@vkT)@A.T@A@vk
      etak = np.linalg.inv(d)@e

      xik= (1/sk)*(Im-uk@ukT)@A@(vk+etak)
      #step4
      uk,vk = alg445._R(uk,vk,etak,xik)
      if param["verbose"]:
        p = Uk.shape[1]
        l = [1/(i+1) for i in range(p)]
        #l = [i+1 for i in range(p)]
        #l.reverse()
        N = np.diag(l)
        N.astype(np.longdouble)

        alg445.log(-np.trace(Uk.T@A@Vk@N),alg445.norm_gradF(A,N,Uk,Vk),np.nan,"log",param)
        print(k,-np.trace(Uk.T@A@Vk@N),alg445.norm_gradF(A,N,Uk,Vk),np.nan)
    #step5
    return uk,vk

  def set_parameter(self):
    param = {}
    param["fn"] = self.fn
    param["verbose"] = self.verbose
    return param

  def solve(self,A,rank,U,V):
    param = self.set_parameter()

    for i in range(rank):
      u0 = U[:,i]
      v0 = V[:,i]
      
      #step2-5
      uk,vk = alg445.run(A,u0,v0,U,V,param)

      U[:,i] = uk.reshape(uk.shape[0],)
      V[:,i] = vk.reshape(vk.shape[0],)
    return uk,vk
  
