# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 22:54:09 2018

@author: srujanponnur
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df=pd.read_csv("15k.csv")
df = df.rename(columns={'satisfaction_level': 'satisfaction', 
                        'last_evaluation': 'evaluation',
                        'number_projects': 'projectCount',
                        'average_monthly_hours': 'avgMnthlyHrs',
                        'time_spent_company': 'yrsAtCmp',
                        'work_accident': 'workAccident',
                        'promotion_last_5_years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })
df['department']=df['department'].astype('category').cat.codes
target='turnover'
x=df.drop('turnover',axis=1)
y=df[target]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state\
=123,stratify=y)
dtree=tree.DecisionTreeClassifier(class_weight='balanced',\
min_weight_fraction_leaf=0.01)
dtree=dtree.fit(x_train,y_train)
importance=dtree.feature_importances_
features=df.drop('turnover',axis=1).columns
indices=np.argsort(importance)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature importance by DecisionTreeClassifier")
plt.bar(range(len(indices)), importance[indices], color='lightblue',  align="center")
plt.step(range(len(indices)), np.cumsum(importance[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), features[indices], rotation=-35,fontsize=12)
plt.xlim([-1, len(indices)])
plt.show()
