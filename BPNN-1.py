import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data = pd.read_csv(r"zzz_1.csv")
data


data.shape


X = data.iloc[ : ,:-1]
y = data.iloc[ : ,-1]
y = np.array(y)
y


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=99)


Xtrain.head()


#恢复索引
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

mpl=GridSearchCV(MLPRegressor( activation = "relu", max_iter = 10000, hidden_layer_sizes = 9),
                    param_grid={
                                "solver": [ "adam", "sgd"]}
                                )  
mpl.fit(Xtrain,Ytrain)
mpl_predict=mpl.predict(Xtest)

np.arange(1,16,1)

mpl.best_params_          

print('R-squared value of BP is',mpl.score(Xtest,Ytest))
print('The mean squared error of BP is',mean_squared_error(ss_Y.inverse_transform(Ytest),
                                                                   ss_Y.inverse_transform(mpl_predict)))
print('The mean absolute error of BP is',mean_absolute_error(ss_Y.inverse_transform(Ytest),
                                                                     ss_Y.inverse_transform(mpl_predict)))
print(' ')


ss_Y.inverse_transform(Ytest)

ss_Y.inverse_transform(mpl_predict)

len(Ytest)

xx=range(0,len(Ytest))
plt.rcParams["font.family"]="SimHei"
plt.figure(figsize=(12,8))
plt.plot(xx,ss_Y.inverse_transform(Ytest),"r-o",label="真实值",linewidth=2)
plt.plot(xx,ss_Y.inverse_transform(mpl_predict),"b:*",label="预测值",linewidth=2)
plt.title("BP神经网络预测输出",fontsize=28)
plt.ylabel("角膜AGEs含量（μg/ml）",fontsize=24)
plt.xlabel("测试样本编号",fontsize=24)
plt.tick_params(labelsize=24)
plt.legend(loc = "lower left",fontsize=20)
fig=plt.gcf()
fig.savefig("BP神经网络预测输出.jpg")
plt.show()
