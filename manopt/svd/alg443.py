import numpy as np
import os
import datetime
import time
from numba import jit

class alg443():
  def __init__(self,fn = None,itemax=5,init="np.svd",c1=1e-30,c2=1e-5,eps=1e-7,VECTOR_TRANSPORT="TP",CALC_BETAKP1="PR",verbose=False):
    if fn == None:
      time = datetime.datetime.now()
      self.fn = time.strftime('%Y%m%d%H%M%S')+".csv"
    else:
      self.fn = fn
    self.init:str = init
    self.c1: float   = c1
    self.c2: float   = c2
    self.eps: float  = eps
    self.itemax:int= itemax
    self.VECTOR_TRANSPORT:str   = VECTOR_TRANSPORT
    self.CALC_BETAKP1:str       = CALC_BETAKP1
    self.verbose:bool = verbose
    self._alphakold = None
  
  @staticmethod
  @jit(cache=True,forceobj=True)
  def initial_point(A,rank,param):
    m = A.shape[0]
    n = A.shape[1]
    p = rank
    
    if param["init"] == "ones":
      U0 = np.ones((m,p))
      V0 = np.ones((n,p))
      U0 = alg443.multiqr(U0)
      V0 = alg443.multiqr(V0)
    elif param["init"] == "np.svd":
      U0,_, V0T = np.linalg.svd(A, full_matrices=True)
      U0 = U0[:,:p]
      V0 = V0T.T[:,:p]
      U0 = alg443.multiqr(U0)
      V0 = alg443.multiqr(V0)
    elif param["init"]=="np.random":
      np.random.seed(1)
      U0 = np.random.normal(size=(m,p))
      V0 = np.random.normal(size=(n,p))
      U0 = alg443.multiqr(U0)
      V0 = alg443.multiqr(V0)
    else:
      pass
    U0.astype(np.longdouble)
    V0.astype(np.longdouble)
    return U0,V0

  @staticmethod
  @jit(cache=True,forceobj=True)
  def gradF(A,N,Uk,Vk):
    xik_  = Uk@alg443.multisym(Uk.T@A@Vk@N)  - A@Vk@N
    etak_ = Vk@alg443.multisym(Vk.T@A.T@Uk@N)- A.T@Uk@N
    return xik_,etak_

  @staticmethod
  @jit(cache=True,forceobj=True)
  def norm_gradF(A,N,Uk,Vk):
    xik_,etak_ = alg443.gradF(A,N,Uk,Vk)
    J1 = alg443.norm(xik_)
    J2 = alg443.norm(etak_)
    return J1+J2

  """
  @staticmethod
  @jit(cache=True)
  def sym(X):
    return (X+X.T)*0.5
  """

  @staticmethod
  @jit(cache=True,forceobj=True)
  def multitransp(A):
    """Vectorized matrix transpose.

    ``A`` is assumed to be an array containing ``M`` matrices, each of which
    has dimension ``N x P``.
    That is, ``A`` is an ``M x N x P`` array. Multitransp then returns an array
    containing the ``M`` matrix transposes of the matrices in ``A``, each of
    which will be ``P x N``.
    """
    if A.ndim == 2:
        return A.T
    return np.transpose(A, (0, 2, 1))

  @staticmethod
  @jit(cache=True,forceobj=True)
  def multisym(A):
    """Vectorized matrix symmetrization.

    Given an array ``A`` of matrices (represented as an array of shape ``(k, n,
    n)``), returns a version of ``A`` with each matrix symmetrized, i.e.,
    every matrix ``A[i]`` satisfies ``A[i] == A[i].T``.
    """
    return 0.5 * (A + alg443.multitransp(A))


  @staticmethod
  @jit(cache=True)
  def rho_skew(X):
    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
        if i<j:
          X[i,j]=-X[j,i]
        elif i==j:
          X[i,j]=0
        elif i>j:
          pass
    return X

  @staticmethod
  @jit(cache=True,forceobj=True)
  def vector_transport(Uk,Vk,xi,eta,zeta,chi,param):
    Q1 = alg443.multiqr(Uk+xi)
    Q2 = alg443.multiqr(Vk+eta)

    if param["VECTOR_TRANSPORT"] == "TR":
      Im = np.eye(Uk.shape[0])
      In = np.eye(Vk.shape[0])
      
      tmp = np.linalg.inv(Q1.T@(Uk+xi))
      trpt_xikp1 = Q1@alg443.rho_skew(Q1.T@zeta@tmp)\
                  +(Im -Q1@Q1.T)@zeta@tmp

      tmp = np.linalg.inv(Q2.T@(Vk+eta))
      trpt_etakp1= Q2@alg443.rho_skew(Q2.T@chi@tmp)\
                  +(In -Q2@Q2.T)@chi@tmp
    
    elif param["VECTOR_TRANSPORT"] == "TP":    
      trpt_xikp1 = zeta - Q1@alg443.multisym(Q1.T@zeta)
      trpt_etakp1= chi  - Q2@alg443.multisym(Q2.T@chi)
    
    return trpt_xikp1,trpt_etakp1
  
  @staticmethod
  @jit(cache=True,forceobj=True)
  def cost(A,N,Uk,Vk):
    return -np.trace(Uk.T@A@Vk@N)

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
  
  """
  @staticmethod
  @jit(cache=True)
  def multiqr(U):
    Q, _ = np.linalg.qr(U)
    return Q
  """

  @staticmethod
  @jit(cache=True,forceobj=True)
  def R(Uk,Vk,etak,xik,alphak):
    if alphak == 0:
      Ukp1,Vkp1 = Uk,Vk
    else:
      Ukp1 = alg443.multiqr(Uk+alphak*xik)
      Vkp1 = alg443.multiqr(Vk+alphak*etak)
    return Ukp1,Vkp1

  @staticmethod
  @jit(cache=True,forceobj=True)
  def calc_betakp1(xikp1_,etakp1_,xik_,etak_,Ukp1,Vkp1,param):
    if param["CALC_BETAKP1"] == "PR":
      zetakp1 = xik_ - Ukp1@alg443.multisym(Ukp1.T@xik_)
      chikp1  = etak_- Vkp1@alg443.multisym(Vkp1.T@etak_)
      tmp= np.trace(xikp1_.T@(xikp1_-zetakp1))+ np.trace(etakp1_.T@(etakp1_-chikp1))
      betakp1=tmp/(np.trace(xik_.T@xik_)  + np.trace(etak_.T@etak_))

    elif param["CALC_BETAKP1"] == "FR":
      tmp = np.trace(xikp1_.T@xikp1_)+np.trace(etakp1_.T@etakp1_)
      betakp1=tmp/(np.trace(xik_.T@xik_)+np.trace(etak_.T@etak_))
    
    return betakp1

  @staticmethod
  def savecsv(d_,logpath):
    with open(logpath,"a") as f:
      np.savetxt(f,d_.reshape(1,d_.size), delimiter=",")
    time.sleep(1e-10)
    return
  
  @staticmethod
  def log(e,n,a,logpath,param):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logpath = os.path.join(BASE_DIR,logpath,param["fn"])
    d_ = np.hstack([e,n,a])
    alg443.savecsv(d_,logpath)
    return
  
  @staticmethod
  @jit(cache=True,forceobj=True)
  def norm(tangent_vector):
    n = np.linalg.norm(tangent_vector)
    return n

  @staticmethod
  @jit(cache=True,forceobj=True)
  def armijo_rule(A,N,Uk,Vk,etak,xik,alphak,param):
    left  = alg443.phi(A,N,Uk,Vk,etak,xik,alphak)
    right = alg443.phi(A,N,Uk,Vk,etak,xik,0)\
          + param["c1"]*alg443.phi_grad(A,N,Uk,Vk,etak,xik,0,param)
    return left<right
  
  @staticmethod
  @jit(cache=True,forceobj=True)
  def wolfe_rule(A,N,Uk,Vk,etak,xik,alphak,param):
    left = alg443.phi_grad(A,N,Uk,Vk,etak,xik,alphak,param)
    right= param["c2"]*alg443.phi_grad(A,N,Uk,Vk,etak,xik,0,param)
    return left>right
  
  @staticmethod
  @jit(cache=True,forceobj=True)
  def check_wolfe_condition(A,N,Uk,Vk,etak,xik,alphak,param):  
    flg1 = alg443.armijo_rule(A,N,Uk,Vk,etak,xik,alphak,param)
    flg2 = alg443.wolfe_rule(A,N,Uk,Vk,etak,xik,alphak,param)
    if flg1 and flg2:
      return alphak
    else:
      return 0
    
  @staticmethod
  @jit(cache=True,forceobj=True)
  def phi(A,N,Uk,Vk,etak,xik,alphak):
    Ukp1,Vkp1 = alg443.R(Uk,Vk,etak,xik,alphak)
    return alg443.cost(A,N,Ukp1,Vkp1)

  @staticmethod
  @jit(cache=True,forceobj=True)
  def phi_grad(A,N,Uk,Vk,etak,xik,alphak,param):
    Ukp1,Vkp1         = alg443.R(Uk,Vk,etak,xik,alphak)
    gradF_Uk,gradF_Vk = alg443.gradF(A,N,Ukp1,Vkp1)
    trpt_xikp1,trpt_etakp1  = alg443.vector_transport(Uk,Vk,alphak*xik,alphak*etak,xik,etak,param)
    return np.trace(gradF_Uk.T@trpt_xikp1)+np.trace(gradF_Vk.T@trpt_etakp1)
  
  @staticmethod
  @jit(cache=True,forceobj=True)
  def zoom(alpha_lo,alpha_hi,A,N,Uk,Vk,etak,xik,param):
    eps = 1e-10
    i = 0
    while True:
      if np.abs(alpha_hi-alpha_lo)<=eps:
        break
      if i > 20:
        break
      alpha_j = (alpha_lo+alpha_hi)/2
      phi_alpha_j      = alg443.phi(A,N,Uk,Vk,etak,xik,alpha_j)
      phi_alpha_lo     = alg443.phi(A,N,Uk,Vk,etak,xik,alpha_lo)
      phi_alpha_0      = alg443.phi(A,N,Uk,Vk,etak,xik,0)
      phi_grad_alpha_j = alg443.phi_grad(A,N,Uk,Vk,etak,xik,alpha_j,param)
      phi_grad_alpha_0 = alg443.phi_grad(A,N,Uk,Vk,etak,xik,0      ,param)

      flg1 = phi_alpha_j\
            >phi_alpha_0+param["c1"]*alpha_j*phi_grad_alpha_j
      flg2 = phi_alpha_j>=phi_alpha_lo
      
      if flg1 or flg2:
        alpha_hi= alpha_j
      else:
        flg3 = np.abs(phi_alpha_j)<=-param["c2"]*phi_grad_alpha_0 
        flg4 = phi_grad_alpha_j*(alpha_hi-alpha_lo)>=0
        
        if flg3:
          return alpha_j

        if flg4:
          alpha_hi=alpha_lo
        
        alpha_lo = alpha_j
      i=i+1
    return alpha_lo

  @staticmethod
  @jit(cache=True,forceobj=True)
  def calc_alpha(alpha_pre,alpha_cur,A,N,Uk,Vk,etak,xik,param):
    phi_alpha_cur = alg443.phi(A,N,Uk,Vk,etak,xik,alpha_cur) 
    phi_alpha_pre = alg443.phi(A,N,Uk,Vk,etak,xik,alpha_pre)
    phi_grad_alpha_cur = alg443.phi_grad(A,N,Uk,Vk,etak,xik,alpha_cur,param)
    phi_grad_alpha_pre = alg443.phi_grad(A,N,Uk,Vk,etak,xik,alpha_pre,param)
    
    tmp= phi_alpha_pre-phi_alpha_cur 
    d1 = phi_grad_alpha_pre\
        +phi_grad_alpha_cur\
        -3*tmp/(alpha_pre-alpha_cur)
    if d1**2-phi_grad_alpha_pre*phi_grad_alpha_cur > 0:
      d2 = np.sign(alpha_cur-alpha_pre)\
        *np.sqrt(d1**2-phi_grad_alpha_pre*phi_grad_alpha_cur)

      tmp = phi_grad_alpha_cur+d2-d1
      tmp = tmp/(phi_grad_alpha_cur-phi_grad_alpha_pre+2*d2)
      alphak = alpha_cur - (alpha_cur-alpha_pre)*tmp
    else:
      alphak = (alpha_cur+alpha_pre)/2
    return alphak

  @staticmethod
  @jit(cache=True,forceobj=True)
  def line_search(A,N,Uk,Vk,etak,xik,param):
    alpha_max = 1
    alpha_min = 1e-5

    alpha_pre = 0
    alpha_cur = 0.7
     
    eps = 1e-10
    i = 0
    while True:
      if abs(alpha_cur-alpha_pre)<=eps:
        break
      if i > 20:
        break
      phi_alpha_cur    = alg443.phi(A,N,Uk,Vk,etak,xik,alpha_cur)
      phi_alpha_pre    = alg443.phi(A,N,Uk,Vk,etak,xik,alpha_pre)
      phi_alpha_0      = alg443.phi(A,N,Uk,Vk,etak,xik,0)
      phi_grad_alpha_0 = alg443.phi_grad(A,N,Uk,Vk,etak,xik,0,param)

      flg1 = phi_alpha_cur > phi_alpha_0 + param["c1"] * alpha_cur *phi_grad_alpha_0
      flg2 = phi_alpha_cur> phi_alpha_pre
      flg3 = i> 0

      if flg1 or (flg2 and flg3):
        return alg443.zoom(alpha_pre,alpha_cur,A,N,Uk,Vk,etak,xik,param)
      
      phi_grad_alpha_cur=alg443.phi_grad(A,N,Uk,Vk,etak,xik,alpha_cur,param)
      flg4 = np.abs(phi_grad_alpha_cur)< -param["c2"]* phi_grad_alpha_0
      if flg4:
        return alpha_cur
      
      flg5 = phi_grad_alpha_cur >= 0
      if flg5:
        return alg443.zoom(alpha_cur,alpha_max,A,N,Uk,Vk,etak,xik,param)
      
      alpha     = alg443.calc_alpha(alpha_pre,alpha_cur,A,N,Uk,Vk,etak,xik,param)
      alpha_new = np.min([np.max([alpha, 2*alpha_cur-alpha_pre])
                         ,alpha_cur+9*(alpha_cur-alpha_pre)])
      alpha_pre = alpha_cur
      alpha_cur = alpha_new
      i = i+1
    return alpha_min

  def wolfe_line_search(self,A,N,Uk,Vk,etak,xik,param):
    if self._alphakold is not None:
      alphak    = alg443.check_wolfe_condition(A,N,Uk,Vk,etak,xik,self._alphakold,param)
      if alphak == 0:
        alphak    = alg443.line_search(A,N,Uk,Vk,etak,xik,param)  
    else:
      alphak    = alg443.line_search(A,N,Uk,Vk,etak,xik,param)
    alphak    = alg443.check_wolfe_condition(A,N,Uk,Vk,etak,xik,alphak,param)
    self._alphakold = alphak
    return alphak

  @staticmethod
  @jit(cache=True,forceobj=True)
  def calc_sacaling(xik,etak,trpt_xikp1,trpt_etakp1):
    tmp = (alg443.norm(xik)+alg443.norm(etak))/(alg443.norm(trpt_xikp1)+alg443.norm(trpt_etakp1))
    Ckp1 = np.min([1,tmp])
    return Ckp1

  def run(self,A,N,Uk,Vk,param):
    #step2
    xik_,etak_ = alg443.gradF(A,N,Uk,Vk)
    xik  = -xik_
    etak = -etak_

    #step3
    c = 0
    while True:
      c = c+1
      if param["verbose"]:
        if c ==1:alphak = np.nan
        alg443.log(np.trace(Uk.T@A@Vk@N),alg443.norm_gradF(A,N,Uk,Vk),alphak,"log",param)
        print(c,np.trace(Uk.T@A@Vk@N),alg443.norm_gradF(A,N,Uk,Vk),alphak)
      
      if alg443.norm_gradF(A,N,Uk,Vk)<param["eps"]:
        break
      
      if c > param["itemax"]:
        break

      #step4
      alphak = self.wolfe_line_search(A,N,Uk,Vk,etak,xik,param)
      Ukp1,Vkp1 = alg443.R(Uk,Vk,etak,xik,alphak)

      #step5
      xikp1_, etakp1_ = alg443.gradF(A,N,Ukp1,Vkp1)
        
      #step6
      betakp1 = alg443.calc_betakp1(xikp1_,etakp1_,xik_,etak_,Ukp1,Vkp1,param)

      #step7
      trpt_xikp1,trpt_etakp1 = alg443.vector_transport(Uk,Vk,alphak*xik,alphak*etak,xik,etak,param)
      Ckp1 = alg443.calc_sacaling(xik,etak,trpt_xikp1,trpt_etakp1)
      xikp1 = -xikp1_ + betakp1*Ckp1*trpt_xikp1
      etakp1= -etakp1_+ betakp1*Ckp1*trpt_etakp1

      #step8
      xik  = xikp1.copy()
      etak = etakp1.copy()
      xik_ = xikp1_.copy()
      etak_= etakp1_.copy()
      Uk   = Ukp1.copy()
      Vk   = Vkp1.copy()
    return Uk,Vk

  def set_parameter(self):
    param = {}
    param["fn"] = self.fn
    param["init"] = self.init
    param["c1"]   = self.c1
    param["c2"]   = self.c2
    param["eps"]  = self.eps
    param["itemax"]= self.itemax
    param["VECTOR_TRANSPORT"]   = self.VECTOR_TRANSPORT
    param["CALC_BETAKP1"]       = self.CALC_BETAKP1
    param["verbose"] = self.verbose
    return param

  def solve(self,A,rank):
    param = self.set_parameter()

    #step1
    U0,V0=alg443.initial_point(A,rank,param)
    
    #l = [i+1 for i in range(rank)]
    #l.reverse()
    l = [1/(i+1) for i in range(rank)]
    N = np.diag(l)
    N.astype(np.longdouble)
        
    #step 3-8
    Uk,Vk = self.run(A,N,U0,V0,param)
    return Uk,Vk
