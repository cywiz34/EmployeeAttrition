# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 00:38:41 2018

@author: srujanponnur
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,precision_recall_curve
from sklearn.metrics import precision_score,recall_score,confusion_matrix,roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve
df=pd.read_csv("15k.csv")
df = df.rename(columns={'satisfaction_level': 'satisfaction', 
                        'last_evaluation': 'evaluation',
                        'number_projects': 'projectCount',
                        'average_monthly_hours': 'averageMonthlyHours',
                        'time_spent_company': 'yearsAtCompany',
                        'work_accident': 'workAccident',
                        'promotion_last_5_years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })

df['department']=df['department'].astype('category').cat.codes
df['salary']=df['salary'].astype('category').cat.codes

front=df['turnover']
df.drop(labels=['turnover'],axis=1,inplace=True)
df.insert(0,'turnover',front)

df['int']=1
indep_var=['satisfaction','evaluation','yearsAtCompany','int','turnover']
df=df[indep_var]
target='turnover'
x=df.drop('turnover',axis=1)
y=df[target]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=123\
,stratify=y)
rf = RandomForestClassifier(
    n_estimators=1000, 
    max_depth=None, 
    min_samples_split=10, 
    class_weight="balanced"
    #min_weight_fraction_leaf=0.02 
    )
rf.fit(x_train, y_train)
print ("\n\n ---Random Forest Model---")
print ("Random Forest-Accuracy is %2.2f"%(accuracy_score(y_test,rf.predict(x_test))))
rf_roc_auc = roc_auc_score(y_test, rf.predict(x_test))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test, rf.predict(x_test)))
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0,1], [0,1],label='Base Rate' 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()
