import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def set_matrix_size(matrix_size):
    df["Matrix Size"] =matrix_size
    return df

def calc_improvement_rate(df):
    df["Improvement Rate"]=(df["norm_conv"]-df["norm_prop"])/df["norm_conv"]*100
    return df

matrix_size = []
df = pd.DataFrame()
for n,i in enumerate(np.logspace(1,3,3,base=10,dtype=np.int)):
    
    i = str(i).zfill(3)
    key = "m_{0}_n_{1}_*.csv".format(i,i)
    csvfnlist = glob.glob(key)

    for j,csvfn in enumerate(csvfnlist):
        df_tmp = pd.read_csv(csvfn)
        matrix_size.append(int(i))
        df = pd.concat([df,df_tmp])

df = set_matrix_size(matrix_size)
df = calc_improvement_rate(df)

boxplot = df.boxplot(column=['Improvement Rate'], by=['Matrix Size'])
plt.savefig("summary0.png")

boxplot = df.boxplot(column=["norm_prop","norm_conv"], by=['Matrix Size'])
plt.savefig("summary1.png")
