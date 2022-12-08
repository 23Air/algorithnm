import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

data = pd.read_csv(r"zzz_4.csv")
data

data.shape

X = data.iloc[ : ,:-1]
y = data.iloc[ : ,-1]
y = np.array(y)
y

i = random.randint(1,100)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=99)
i

Xtrain.head()

for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])

Xtrain.head()

ss_X = StandardScaler()
ss_Y = StandardScaler()
Xtrain = ss_X.fit_transform(Xtrain)
Xtest = ss_X.transform(Xtest)
Ytrain = ss_Y.fit_transform(Ytrain.reshape(-1,1))
Ytest = ss_Y.transform(Ytest.reshape(-1,1))

Ytest.shape


poly_svr=GridSearchCV(SVR(kernel="poly"),     
                    param_grid={"C":[1e0, 1e1, 1e2, 1e3],
                                "gamma": ["scale", "auto"],
                                "degree": np.arange(0,10,1)}) 
poly_svr.fit(Xtrain,Ytrain)
poly_svr_y_predict=poly_svr.predict(Xtest)


a = np.arange(0,10,1)
a

poly_svr.best_params_           

print('R-squared value of linear SVR is',poly_svr.score(Xtest,Ytest))
print('The mean squared error of linear SVR is',mean_squared_error(ss_Y.inverse_transform(Ytest),
                                                                   ss_Y.inverse_transform(poly_svr_y_predict)))
print('The mean absolute error of linear SVR is',mean_absolute_error(ss_Y.inverse_transform(Ytest),
                                                                     ss_Y.inverse_transform(poly_svr_y_predict)))
print(' ')

ss_Y.inverse_transform(Ytest)

ss_Y.inverse_transform(poly_svr_y_predict)

