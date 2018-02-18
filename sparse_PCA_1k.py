import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import decomposition


df = pd.DataFrame.from_csv('data_1k_num.csv', index_col=None)


jb=df.JobRole.unique()
#print jb
for i in range(len(jb)):
	df.JobRole.replace(jb[i],i+1,inplace=True)

dept=df.Department.unique()
for i in range(len(dept)):
	df.Department.replace(dept[i],i+1,inplace=True)

est=decomposition.SparsePCA(20, alpha=0.5,max_iter=100)

c=est.fit(df)
df1=pd.DataFrame(est.components_)
df1=(df1)
df1.to_csv("Reduced_Components.csv")
print est.components_
