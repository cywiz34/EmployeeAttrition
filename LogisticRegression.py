# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 18:58:02 2018

@author: srujanponnur
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,precision_recall_curve
from sklearn.metrics import precision_score,recall_score,confusion_matrix,roc_auc_score
from sklearn.preprocessing import RobustScaler
import statsmodels.api as sm
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
indep_var=indep_var[:-1]
logreg=sm.Logit(y_train,x_train[indep_var])
answer=logreg.fit()
print answer.params
coeff=answer.params
"""var=['satisfaction','evaluation','yearsAtCompany']
df1=df[var]
calculating employee turnover score using logistic regression
def calc_theta(coeff,satisfaction,evaluation,yearsatcompany):
    return coeff[3]+coeff[0]*satisfaction#+coeff[1]*evaluation+coeff[2]*yearsatcompany
for row in df1:    
    val=calc_theta(coeff,row[0],row[1],row[2])
    p = np.exp(val)/(1+np.exp(val))    
    score.append(p)"""
model=LogisticRegression(penalty='l2',C=1)
model.fit(x_train,y_train)
print ("\n\n ---Logistic Model---\n")
print ("Logistic Accuracy is %2.2f"%(accuracy_score(y_test,model.predict(x_test))))
logis = LogisticRegression(class_weight = "balanced")
logis.fit(x_train, y_train)
logit_roc_auc = roc_auc_score(y_test, logis.predict(x_test))
print ("Logistic AUC = %2.2f" % logit_roc_auc)
print(classification_report(y_test, logis.predict(x_test)))
fpr,tpr,frequency=roc_curve(y_test,logis.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label="logistic Regression(area=%0.2f)"%logit_roc_auc)
plt.plot([0,1], [0,1],label='Base Rate' 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()







