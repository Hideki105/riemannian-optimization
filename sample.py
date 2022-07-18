import numpy as np
eps: float = 1e-14
init:str = "np.random"
#0<c1<c2<1
c1: float=1e-4
c2: float=0.1

np.random.seed(100)
m, n, rank = 500, 300, 10
Ur, _ = np.linalg.qr(np.random.normal(size=(m, rank)))
Vr, _ = np.linalg.qr(np.random.normal(size=(rank, n)))
S = np.diag(np.arange(rank, 0, -1))
A = Ur @ S @ Vr.T
A.astype(np.longdouble)

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

time = datetime.datetime.now()
csvfn = time.strftime('%Y%m%d%H%M%S')+".csv"
U,V = svd(A,rank,init)
S = set_sigma(A,U,V)
viewer.bar(A,U,V,rank,csvfn)

from manopt.svd.svd import svd
from manopt.svd.alg445 import alg445
from manopt.svd.util   import parameter,evaluation
from manopt.svd.util   import set_sigma
import manopt.svd.viewer as viewer

Uinit,Vinit = svd(A,rank,init)

alg = alg445(verbose=True)
U,V = alg.solve(A,rank,Uinit,Vinit)
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
