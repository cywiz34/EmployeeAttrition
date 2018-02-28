import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
#matplotlib inline


df = pd.DataFrame.from_csv('income_job.csv', index_col=None)



#print df.describe()


income_job = df.groupby('JobLevel')
temp=np.array(income_job.mean())

data=[]
for i in np.array(df):
    data.append([i[0],i[1],(i[1]-temp[i[0]-1][0])/temp[i[0]-1][0],i[1]-temp[i[0]-1][0]])

data=pd.DataFrame(data=data)


data.to_csv("salary_mean.csv")
