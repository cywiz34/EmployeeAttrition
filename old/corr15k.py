import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
#matplotlib inline


df = pd.DataFrame.from_csv('data_15k.csv', index_col=None)

df = df.rename(columns={'satisfaction_level': 'satisfaction', 
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'department' : 'department',
                        'left' : 'turnover'
                        })


front = df['turnover']
df.drop(labels=['turnover'], axis=1,inplace = True)
df.insert(0, 'turnover', front)
df.head()

#print df.dtypes

turnover_rate = df.turnover.value_counts() / len(df)
#print turnover_rate
#print df.describe()


turnover_Summary = df.groupby('turnover')
#turnover_Summary.mean()

corr = df.corr()
corr = (corr)
corr.to_csv("15_corr.csv")
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values)

plt.show()
