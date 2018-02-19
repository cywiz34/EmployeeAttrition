from sklearn.model_selection import train_test_split
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import time
# define example
cate=[]#to store encoded categorical values
col_names=[]# to store category names
dfc = pd.DataFrame.from_csv('1k_categorical.csv', index_col=None)
val=array(dfc)
<<<<<<< HEAD
#print list(dfc)
#print val.T
=======
print list(dfc)
print val.T
>>>>>>> 3b51054dbb94d488ea88d7ed9dce746d88ec4740
for data in val.T:
    value = data
    # integer encode changing labes to integers
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(value)
<<<<<<< HEAD
    #print list(label_encoder.classes_)
=======
    print list(label_encoder.classes_)
>>>>>>> 3b51054dbb94d488ea88d7ed9dce746d88ec4740
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
<<<<<<< HEAD


=======
>>>>>>> 3b51054dbb94d488ea88d7ed9dce746d88ec4740
#print col_names
#print array(cate).T
#categorical DataFrame
dfc=pd.DataFrame(data=array(cate).T,columns=col_names)
#non categorical DataFrame
dfnc=pd.DataFrame.from_csv('1k_noncate.csv', index_col=None)
#combining both data frames
df=pd.concat([dfnc,dfc],axis=1)
#df.to_csv("combined.csv")
<<<<<<< HEAD

print df['left'].value_counts()

#X is our data variable and y is our target variable
#X, y = array(df).T[1:],array(df).T[0]

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
=======
>>>>>>> 3b51054dbb94d488ea88d7ed9dce746d88ec4740
