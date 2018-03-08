from sklearn.model_selection import train_test_split
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import pandas_ml as pdml
import time
# define example
cate=[]#to store encoded categorical values
col_names=[]# to store category names
dfc = pd.DataFrame.from_csv('1k_categorical.csv', index_col=None)
val=array(dfc)
#print list(dfc)
#print val.T
for data in val.T:
    value = data
    # integer encode changing labes to integers
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(value)
    #print list(label_encoder.classes_)
    col_names+=list(label_encoder.classes_)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    #reshaping integer encoded values to column vectors
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    #perform one-hot encoding
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(onehot_encoded.T)
    for col in onehot_encoded.T:
        cate.append(col)
    #time.sleep(1)


#print col_names
#print array(cate).T
#categorical DataFrame
dfc=pd.DataFrame(data=array(cate).T,columns=col_names)
#non categorical DataFrame
dfnc=pd.DataFrame.from_csv('1k_noncate.csv', index_col=None)
#combining both data frames
df=pd.concat([dfnc,dfc],axis=1)
#df.to_csv("combined.csv")

print df['Attrition'].value_counts()

train, test = train_test_split(df, test_size=0.2)
#print train.describe()
X_train=train.drop(['Attrition'],axis=1)
X_test=test.drop(['Attrition'],axis=1)
Y_train=train.pop('Attrition')
Y_test=test.pop('Attrition')
#print len(Y_test),len(Y_train)
'''for i in X_train:
    for j in range(len(array(X_train[i]))):
        array(X_train[i])[j]=(array(X_train[i])[j]-min(X_train[i]))/(max(X_train[i])-min(X_train[i]))
print X_train'''
