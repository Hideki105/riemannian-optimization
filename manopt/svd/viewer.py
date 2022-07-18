import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from manopt.svd.util import evaluation
import matplotlib
matplotlib.use('Agg')

def read_csv(csvfn):
  return pd.read_csv(csvfn,header=None)

def plotter(x1,x2,x3,pngfn):
  fig = plt.figure()
  ax1 = fig.add_subplot(2, 2, 1)
  ax2 = fig.add_subplot(2, 2, 2)
  ax3 = fig.add_subplot(2, 2, 3)
  
  ax1.plot(x1);
  ax1.set_xlabel("iteration")
  ax1.set_ylabel("cost");
  ax1.grid()

  ax2.plot(x2);
  ax2.set_xlabel("iteration")
  ax2.set_ylabel("norm");
  ax2.set_yscale('log')
  ax2.grid()

  ax3.plot(x3);
  ax3.set_xlabel("iteration")
  ax3.set_ylabel(r"$\alpha_{k}$");
  ax3.set_yscale('log')
  plt.tight_layout()
  ax3.grid()
  plt.savefig(pngfn)

def run(csvfn,logpath="log"):
  BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  csvfn = os.path.join(BASE_DIR,logpath,csvfn)
  df = read_csv(csvfn)
  x1 = df.iloc[:,0]
  x2 = df.iloc[:,1]
  x3 = df.iloc[:,2]
  pngfn = csvfn.replace(".csv",".png")
  plotter(x1,x2,x3,pngfn)
    
def bar(A,U,V,rank,csvfn,logpath="log"):
  BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  csvfn = os.path.join(BASE_DIR,logpath,csvfn)
  pngfn = csvfn.replace(".csv","_bar.png")

  eval = evaluation(A,U,V,rank,csvfn)
  eval_result = eval.result()
  eval.savecsv()

  norm_conv = eval_result["norm_conv"][0]
  norm_prop = eval_result["norm_prop"][0]
  conv_IU0  = eval_result["norm_conv_IU0"][0]
  prop_IU   = eval_result["norm_prop_IU"][0]
  conv_IV0  = eval_result["norm_conv_IV0"][0]
  prop_IV   = eval_result["norm_prop_IV"][0]
  
  fig = plt.figure(figsize=(18,6))
  ax1 = fig.add_subplot(1, 3, 1)
  ax2 = fig.add_subplot(1, 3, 2)
  ax3 = fig.add_subplot(1, 3, 3)
  
  height= np.array([norm_conv,norm_prop])
  label = ["Numpy", "Riemannian Optimization"]
  ax1.bar(x=[1,2],height=height, tick_label=label, align="center")
  ax1.grid()

  height= np.array([conv_IU0,prop_IU])
  label = ["Numpy", "Riemannian Optimization"]
  ax2.bar(x=[1,2],height=height, tick_label=label, align="center")
  ax2.grid()

  height= np.array([conv_IV0,prop_IV])
  label = ["Numpy", "Riemannian Optimization"]
  ax3.bar(x=[1,2],height=height, tick_label=label, align="center")
  ax3.grid()
  
  plt.tight_layout()
  plt.savefig(pngfn)

