from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import random
import time
import xgboost as xgb
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.metrics import r2_score
import joblib
from sklearn.model_selection import TimeSeriesSplit  
from sklearn.preprocessing import StandardScaler     

data = pd.read_csv(r"zzz_1.csv")
data

name = ["角膜AGEs荧光强度值","角膜AGEs浓度值"]
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


i = random.randint(1,100)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=99)
i

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
ss_X = StandardScaler()
ss_Y = StandardScaler()

from sklearn import preprocessing
from sklearn.metrics import r2_score
tscv = TimeSeriesSplit(n_splits=5)
def plotModelResults(model, X_train, X_test, y_train, y_test, plot_intervals=False, plot_anomalies=False):
    """
        Plots modelled vs fact values, prediction intervals and anomalies
    """
    prediction = model.predict(X_test)
 
    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test, label="actual", linewidth=2.0)
 
    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train,
                             cv=tscv,
                             scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()
 
        scale = 20
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)
 
        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
 
        if plot_anomalies:
            anomalies = np.array([np.NaN] * len(y_test))
            anomalies[y_test < lower] = y_test[y_test < lower]
            anomalies[y_test > upper] = y_test[y_test > upper]
            plt.plot(anomalies, "o", markersize=10, label="Anomalies")
 
  #  error = mean_absolute_percentage_error(prediction, y_test)
    error =np.mean(np.abs((prediction -  y_test) /prediction)) * 100
    plt.title("Mean absolute percentage error {0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);
    plt.savefig("linear.png")


def model_fit_regressor(x_train,x_test,y_train,y_test):
    model = xgb.XGBRegressor(max_depth=10, learning_rate=0.1, n_estimators=1000, reg_alpha=0.005, subsample=0.8,
                             gamma=0,colsample_bylevel=0.8, objective ='reg:squarederror')
    scaler = StandardScaler()
    columns = x_train.columns
    indexs_train = x_train.index
    x_train = pd.DataFrame(scaler.fit_transform(x_train),index = indexs_train, columns = columns)
    indexs_test = x_test.index
    x_test = pd.DataFrame(scaler.transform(x_test),index = indexs_test, columns = columns)


    
    
    model.fit(x_train, y_train) 
   
    score = model.score(x_train, y_train)   
    print("Training score: ", score) 
    
  
    # - cross validataion 
    scores = cross_val_score(model, x_train, y_train, cv=5)
    print("Mean cross-validation score: %.2f" % scores.mean())
    
    kfold = KFold(n_splits=10, shuffle=True)
    kf_cv_scores = cross_val_score(model, x_train, y_train, cv=kfold )
    print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
     
    ypred = model.predict(x_test)
    mse = mean_squared_error(y_test, ypred)
    print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % (mse**(1/2.0)))
    
    reg = lr().fit(x_test, y_test)
    r2= reg.score(x_test, y_test)
    print("r2: ", r2) 
    
    #x_ax = range(len(y_test))
    #plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
    ##plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
    #plt.legend()
    #plt.show()
    
    plotModelResults(model, X_train=x_train, X_test=x_test,  y_train=y_train, y_test=y_test, plot_intervals=True, plot_anomalies=True)
    joblib.dump(model, 'OilCan.pkl')
    
    return model,y_test,ypred

model,y_test,ypred=model_fit_regressor(Xtrain,Xtest,Ytrain,Ytest)

xx=range(0,len(Ytest))
plt.figure(figsize=(12,6))
plt.plot(xx,y_test,color="b",label="Ture",linewidth=2) 
plt.plot(xx,ypred,color="r",label="Predict",linewidth=2)
plt.legend()
fig=plt.gcf()
fig.savefig("./XGBoost.jpg")
plt.show()

y_test

ypred

df = pd.DataFrame([y_test,ypred])

df.to_csv("./data.csv", index=False)
