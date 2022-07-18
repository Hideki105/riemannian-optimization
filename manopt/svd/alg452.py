import numpy as np
import os
import datetime
import time

from manopt.svd.alg443 import alg443
from manopt.svd.alg445 import alg445

class alg452():
  def __init__(self,fn = None,init="np.svd",c1=1e-30,c2=1e-5,eps=1e-9,VECTOR_TRANSPORT="TP",CALC_BETAKP1="PR",verbose=False):
    if fn == None:
      time = datetime.datetime.now()
      self.fn = time.strftime('%Y%m%d%H%M%S')+".csv"
    else:
      self.fn = fn
    self.init = init
    self.c1   = c1
    self.c2   = c2
    self.eps  = eps
    self.VECTOR_TRANSPORT   = VECTOR_TRANSPORT
    self.CALC_BETAKP1       = CALC_BETAKP1
    self.verbose = verbose
  

  def solve(self,A,rank):
    alg1 = alg443(fn=self.fn, init=self.init,c1=self.c1,c2=self.c2,eps=self.eps,VECTOR_TRANSPORT=self.VECTOR_TRANSPORT,CALC_BETAKP1=self.CALC_BETAKP1,verbose=self.verbose)
    param = alg1.set_parameter()
    Uk,Vk=alg1.initial_point(A,rank,param)
    
    
    #l = [i+1 for i in range(rank)]
    #l.reverse()
    l = [1/(i+1) for i in range(rank)]
    N = np.diag(l)
    N.astype(np.longdouble)

    #step 4-7
    Uk,Vk = alg1.run(A,N,Uk,Vk,param)
    
    U = Uk
    V = Vk
    
    for i in range(rank):
      u0 = U[:,i]
      v0 = V[:,i]
      
      #step2-5
      alg2 = alg445(fn=alg1.fn,verbose=self.verbose)
      uk,vk = alg2.run(A,u0,v0,U,V,param)

      U[:,i] = uk.reshape(uk.shape[0],)
      V[:,i] = vk.reshape(vk.shape[0],)
    return U,V
  

