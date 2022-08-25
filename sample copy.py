import numpy as np
import os
import time
eps: float = 1e-14
init:str = "np.svd"
#0<c1<c2<1
c1: float=1e-4
c2: float=0.1

np.random.seed(100)
m, n, rank = 1000, 1000, 1000
Ur, _ = np.linalg.qr(np.random.normal(size=(m, rank)))
Vr, _ = np.linalg.qr(np.random.normal(size=(rank, n)))
S = np.diag(np.arange(rank, 0, -1))
A = Ur @ S @ Vr.T
A.astype(np.longdouble)


from manopt.svd.svd import svd
import numpy as np
from   manopt.svd.util   import set_sigma
import manopt.svd.viewer as viewer
import datetime

t_start = time.time()
time_ = datetime.datetime.now()
csvfn = time_.strftime('%Y%m%d%H%M%S')+".csv"
U,V,res = svd(A,rank,init)
t_end = time.time()

print(t_end-t_start)