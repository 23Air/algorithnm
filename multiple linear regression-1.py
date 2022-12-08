from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

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

cross_val_score(reg,X,y,cv=10,scoring="r2").mean()
Ytest
yhat

df = pd.DataFrame([Ytest.transpose(),yhat.transpose()])
df.to_csv("./data.csv", index=False)

xx=range(0,len(Ytest))
plt.rcParams["font.family"]="SimHei"
plt.figure(figsize=(12,8))
# plt.plot(xx,ss_Y.inverse_transform(Ytest),color="b",label="Ture",linewidth=2) 
plt.plot(xx,Ytest,"r-o",label="真实值",linewidth=2)
# plt.plot(xx,ss_Y.inverse_transform(mpl_predict),color="r",label="Predict",linewidth=2)
plt.plot(xx,yhat,"b:*",label="预测值",linewidth=2)
plt.title("多元线性回归预测输出",fontsize=28)
plt.ylabel("角膜AGEs含量（μg/ml）",fontsize=24)
plt.xlabel("测试样本编号",fontsize=24)
plt.tick_params(labelsize=24)
plt.legend(loc = "lower left",fontsize=20)
fig=plt.gcf()
fig.savefig("多元线性回归预测输出.jpg")
plt.show()

