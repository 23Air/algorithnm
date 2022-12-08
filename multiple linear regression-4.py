from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

data = pd.read_csv(r"zzz_4.csv")
data

name = ["糖化血红蛋白","空腹血糖","患病时长","角膜AGEs荧光强度值","角膜AGEs浓度值"]
data.columns = name 
data

data.shape

X = data.iloc[ : ,:-1]
X

y = data.iloc[ : ,-1]
y = np.array(y)
y

y.shape

y.min()

y.max()

# Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=66)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3)


Xtrain.head()


Xtest.head()


Ytrain

Ytrain.shape


Ytest


Ytest.shape

for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])

Xtrain.head()

Xtest.head()

Xtrain.shape

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain_scaled = scaler.transform(Xtrain)
Xtrain_scaled
Xtest_scaled = scaler.transform(Xtest)
reg = lr().fit(Xtrain, Ytrain)


yhat = reg.predict(Xtest) 
yhat

reg.coef_ 

Xtrain.columns

[*zip(Xtrain.columns,reg.coef_)]

reg.intercept_

from sklearn.metrics import mean_squared_error as MSE
MSE(Ytest,yhat)

Ytest.mean()

cross_val_score(reg,X,y,cv=10,scoring="neg_mean_squared_error").mean()


from sklearn.metrics import r2_score
r2_score(yhat,Ytest)  

r2 = reg.score(Xtest,Ytest)
r2

r2_score(Ytest,yhat)


r2_score(y_true = Ytest,y_pred = yhat)

import matplotlib.pyplot as plt
sorted(Ytest)  


# In[39]:


plt.plot(range(len(Ytest)),sorted(Ytest),c="black",label= "Data")
plt.plot(range(len(yhat)),sorted(yhat),c="red",label = "Predict")
plt.legend()
plt.show()

