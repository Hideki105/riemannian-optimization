import numpy as np
import os
import time
eps: float = 1e-14
init:str = "np.random"
#0<c1<c2<1
c1: float=1e-4
c2: float=0.1

np.random.seed(100)
m, n, rank = 5, 3, 2
Ur, _ = np.linalg.qr(np.random.normal(size=(m, rank)))
Vr, _ = np.linalg.qr(np.random.normal(size=(rank, n)))
S = np.diag(np.arange(rank, 0, -1))
A = Ur @ S @ Vr.T
A.astype(np.longdouble)

from manopt.svd.svd_hybrid import svd_hybrid
import numpy as np
from   manopt.svd.util   import set_sigma
import manopt.svd.viewer as viewer
import datetime
from manopt.svd.util   import evaluation

def to_csv(csvfn,res):
    logpath = ".\\manopt\\log"
    csvfn = os.path.join(logpath,csvfn)
    cost     = np.array(res.log["iterations"]["cost"])
    gradnorm = np.array(res.log["iterations"]["gradient_norm"])
    n = len(cost)
    alhpak   = np.array([np.nan for i in range(n)])
    data = np.hstack([cost.reshape(n,1),gradnorm.reshape(n,1),alhpak.reshape(n,1)])
    np.savetxt(csvfn,data, delimiter =',')

# BetaTypes of the conjugate gradient method in pymanopt was changed.
BetaTypes = ["DaiYuan","PolakRibiere", "Hybrid1", "Hybrid2"]

J= []
for beta_type in BetaTypes:
    time_ = datetime.datetime.now()
    csvfn = time_.strftime('%Y%m%d%H%M%S')+".csv"
    U,V,res = svd_hybrid(A,rank,beta_type,init)
    S = set_sigma(A,U,V)
    A_= U@S@V.T
    to_csv(csvfn,res)
    viewer.bar(A,U,V,rank,csvfn)
    viewer.run(csvfn)
    eval = evaluation(A,U,V,rank,csvfn)
    eval_result = eval.result()
    eval.savecsv()
    J.append(np.linalg.norm(A-A_))
    time.sleep(1)

from manopt.svd.alg445 import alg445
from manopt.svd.util   import parameter,evaluation
from manopt.svd.util   import set_sigma
import manopt.svd.viewer as viewer
from manopt.svd.svd_hybrid import svd_hybrid
import datetime

beta_type = BetaTypes[np.argmin(J)]
print(beta_type)
time_ = datetime.datetime.now()
csvfn = time_.strftime('%Y%m%d%H%M%S')+".csv"
U,V,res = svd_hybrid(A,rank,beta_type,init)
to_csv(csvfn,res)
alg = alg445(verbose=True)
alg.fn = csvfn
U,V = alg.solve(A,rank,U,V)
S = set_sigma(A,U,V)
A_= U@S@V.T
viewer.bar(A,U,V,rank,csvfn)
viewer.run(csvfn)
eval = evaluation(A,U,V,rank,csvfn)
eval_result = eval.result()
eval.savecsv()
time.sleep(1)

from manopt.svd.alg441 import alg441
from manopt.svd.util   import parameter,evaluation
from manopt.svd.util   import set_sigma
import manopt.svd.viewer as viewer

alg = alg441(init=init,c1=c1,eps=eps,verbose=True)
U,V = alg.solve(A,rank)
S = set_sigma(A,U,V)
A_= U@S@V.T
print(A)
print(A_)
viewer.run(alg.fn)
viewer.bar(A,U,V,rank,alg.fn)
parameter(alg.fn,init,c1,np.nan,np.nan,np.nan,eps).save()
eval = evaluation(A,U,V,rank,alg.fn)
eval_result = eval.result()
eval.savecsv()

from manopt.svd.alg443 import alg443
from manopt.svd.util   import parameter,evaluation
from manopt.svd.util   import set_sigma
import manopt.svd.viewer as viewer

J = []
VT=[]
CB=[]

for vt in ["TP","TR"]:
    for cb in ["PR","FR"]:
        print(vt,cb)
        alg = alg443(init=init,c1=c1,c2=c2,eps=eps,VECTOR_TRANSPORT=vt,CALC_BETAKP1=cb,verbose=True)
        U,V = alg.solve(A,rank)
        S = set_sigma(A,U,V)
        A_= U@S@V.T
        print(A)
        print(A_)
        viewer.run(alg.fn)
        viewer.bar(A,U,V,rank,alg.fn)
        parameter(alg.fn,init,c1,c2,vt,cb,eps).save()
        eval = evaluation(A,U,V,rank,alg.fn)
        eval_result = eval.result()
        eval.savecsv()
        J.append(np.linalg.norm(A-A_))
        VT.append(vt)
        CB.append(cb)

from manopt.svd.alg452 import alg452
from manopt.svd.util   import parameter,evaluation
from manopt.svd.util   import set_sigma
import manopt.svd.viewer as viewer

vt = VT[np.argmin(J)]
cb = CB[np.argmin(J)]

alg = alg452(init=init,c1=c1,c2=c2,eps=eps,VECTOR_TRANSPORT=vt,CALC_BETAKP1=cb,verbose=True)
U,V = alg.solve(A,rank)
S = set_sigma(A,U,V)
A_= U@S@V.T
print(A)
print(A_)
viewer.run(alg.fn)
viewer.bar(A,U,V,rank,alg.fn)
parameter(alg.fn,init,c1,c2,vt,cb,eps).save()
eval = evaluation(A,U,V,rank,alg.fn)
eval_result = eval.result()
eval.savecsv()

from manopt.svd.svd import svd
import numpy as np
from   manopt.svd.util   import set_sigma
import manopt.svd.viewer as viewer
import datetime

time_ = datetime.datetime.now()
csvfn = time_.strftime('%Y%m%d%H%M%S')+".csv"
U,V,res = svd(A,rank,init)
S = set_sigma(A,U,V)
viewer.bar(A,U,V,rank,csvfn)


from manopt.svd.svd import svd
from manopt.svd.alg445 import alg445
from manopt.svd.util   import parameter,evaluation
from manopt.svd.util   import set_sigma
import manopt.svd.viewer as viewer

U,_,VT = np.linalg.svd(A,full_matrices=False) 
U, _ = np.linalg.qr(U[:,:rank])
V, _ = np.linalg.qr(VT.T[:,:rank])

alg = alg445(verbose=True)
U,V = alg.solve(A,rank,U,V)
S = set_sigma(A,U,V)
A_= U@S@V.T
print(A)
print(A_)
viewer.run(alg.fn)
viewer.bar(A,U,V,rank,alg.fn)
parameter(alg.fn,init,c1,np.nan,np.nan,np.nan,eps).save()
eval = evaluation(A,U,V,rank,alg.fn)
eval_result = eval.result()
eval.savecsv()