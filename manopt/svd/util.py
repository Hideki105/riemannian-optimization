import numpy as np
import pandas as pd
import os

def set_sigma(A,U,V):
  S = U.T@A@V
  for i in range(S.shape[0]):
    for j in range(S.shape[1]):
      if i == j:
        pass
      else:
        S[i,j] = 0
  return S

class parameter():
  def __init__(self,csvfn,init,c1,c2,vt,cb,eps):
    self.csvfn = csvfn
    self.init = init
    self.c1 = c1
    self.c2 = c2
    self.vt = vt
    self.cb = cb
    self.eps= eps

  def set_DataFrame(self):
    df = pd.DataFrame()
    if np.isscalar(self.c1):
      df["init"]=[self.init]
      df["c1"] = [self.c1]
      df["c2"] = [self.c2]
      df["vt"] = [self.vt]
      df["cb"] = [self.cb]
      df["eps"] = [self.eps]
    else:
      df["init"]=self.init
      df["c1"] = self.c1
      df["c2"] = self.c2
      df["vt"] = self.vt
      df["cb"] = self.cb
      df["eps"] = self.eps
    return df

  def save(self,logpath="log"):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csvfn = os.path.join(BASE_DIR,logpath,self.csvfn)
    csvfn = csvfn.replace(".csv","_param.csv")

    df = self.set_DataFrame()
    df.to_csv(csvfn)

class evaluation():
  def __init__(self,A,U,V,rank,csvfn):
    self.A = A
    self.U = U
    self.V = V
    self.rank = rank
    self.csvfn = csvfn
  
  def set_Xprop(self):
    S = set_sigma(self.A,self.U,self.V)
    X_prop = self.U@S@self.V.T
    return X_prop,self.U,self.V

  def set_Xconv(self):
    U0, S0, V0T = np.linalg.svd(self.A, full_matrices=False)
    X_conv = U0[:,:self.rank]@np.diag(S0[:self.rank])@V0T[:self.rank,:]
    return X_conv,U0,V0T.T
  
  def savecsv(self,logpath="log"):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
    csvfn = os.path.join(BASE_DIR,logpath,self.csvfn)
    csvfn = csvfn.replace(".csv","_bar.csv")
    df = pd.DataFrame(self.eval_result)
    df.to_csv(csvfn)

  def result(self):
    eval_result = {}
    X_prop,U,V   = self.set_Xprop()
    X_conv,U0,V0 = self.set_Xconv()
    eval_result["norm_prop"] = [np.linalg.norm(self.A-X_prop)]
    eval_result["norm_conv"] = [np.linalg.norm(self.A-X_conv)]

    UTU = U.T@U
    VTV = V.T@V  
    IU = np.eye(UTU.shape[0])
    IV = np.eye(VTV.shape[0])

    eval_result["norm_prop_IU"] = [np.linalg.norm(IU-UTU)]
    eval_result["norm_prop_IV"] = [np.linalg.norm(IV-VTV)]

    U0TU0 = U0.T@U0
    V0TV0 = V0.T@V0
    IU0 = np.eye(U0TU0.shape[0])
    IV0 = np.eye(V0TV0.shape[0])

    eval_result["norm_conv_IU0"]= [np.linalg.norm(IU0-U0TU0)]
    eval_result["norm_conv_IV0"] = [np.linalg.norm(IV0-V0TV0)]

    self.eval_result = eval_result
    return eval_result