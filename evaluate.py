import numpy as np
eps = 1e-100
init = "np.svd"
#0<c1<c2<1
c1=1e-4
c2=1e-1

from manopt.svd.svd import svd
import numpy as np
from   manopt.svd.util   import set_sigma
import manopt.svd.viewer as viewer

def set_csvfn(c,A):
    p1 = str(c).zfill(4)
    p2 = str(A.shape[0]).zfill(3)
    p3 = str(A.shape[1]).zfill(3)
    csvfn = "m_{1}_n_{2}_counter_{0}.csv".format(p1,p2,p3)
    return csvfn

np.random.seed(1)
for i in np.logspace(1,4,4,base=10,dtype=np.int):
    m,n,rank= i,i,i
    init = "np.svd"
    for j in range(100):
        A = np.random.uniform(-10,10,size=(m,n))
        csvfn = set_csvfn(j,A)
        U,V = svd(A,rank,init)
        S = set_sigma(A,U,V)
        viewer.bar(A,U,V,rank,csvfn)
