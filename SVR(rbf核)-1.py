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

data = pd.read_csv(r"zzz_1.csv")
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
Ytest

for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])

Xtrain.head()

ss_X = StandardScaler()
ss_Y = StandardScaler()
Xtrain = ss_X.fit_transform(Xtrain)
Xtest = ss_X.transform(Xtest)
Ytrain = ss_Y.fit_transform(Ytrain.reshape(-1,1))
Ytest = ss_Y.transform(Ytest.reshape(-1,1))

Xtrain

Ytest.shape

rbf_svr=GridSearchCV(SVR(kernel="rbf"),
                    param_grid={"C":[1e0, 1e1, 1e2, 1e3],
                                "gamma": ["scale", "auto"]}
                                )   
rbf_svr.fit(Xtrain,Ytrain)
rbf_svr_y_predict=rbf_svr.predict(Xtest).reshape(-1, 1)


rbf_svr.best_params_           

a = np.logspace(-2, 2, 5)
a


print('R-squared value of linear SVR is',rbf_svr.score(Xtest,Ytest))
print('The mean squared error of linear SVR is',mean_squared_error(ss_Y.inverse_transform(Ytest),
                                                                   ss_Y.inverse_transform(rbf_svr_y_predict)))
print('The mean absolute error of linear SVR is',mean_absolute_error(ss_Y.inverse_transform(Ytest),
                                                                     ss_Y.inverse_transform(rbf_svr_y_predict)))
print(' ')

ss_Y.inverse_transform(Ytest)
ss_Y.inverse_transform(rbf_svr_y_predict)
df = pd.DataFrame(ss_Y.inverse_transform(rbf_svr_y_predict))
df.to_csv("./svrdata.csv", index=False)

xx=range(0,len(Ytest))
plt.rcParams["font.family"]="SimHei"
plt.figure(figsize=(12,8))
plt.plot(xx,ss_Y.inverse_transform(Ytest),"r-o",label="真实值",linewidth=2)
plt.plot(xx,ss_Y.inverse_transform(rbf_svr_y_predict),"b:*",label="预测值",linewidth=2)
plt.title("支持向量回归预测输出",fontsize=28)
plt.ylabel("角膜AGEs含量（μg/ml）",fontsize=24)
plt.xlabel("测试样本编号",fontsize=24)
plt.tick_params(labelsize=24)
plt.legend(loc = "lower left",fontsize=20)
fig=plt.gcf()
fig.savefig("支持向量回归预测输出.jpg")
plt.show()

Ytest


