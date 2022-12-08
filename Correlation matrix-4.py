import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings(action = "once") 

large = 22; med = 16; small = 12
params = {"axes.titlesize": large,
          "legend.fontsize": med,
          "figure.figsize": (16, 10),
          "axes.labelsize": med,
          "axes.titlesize": med,
          "xtick.labelsize": med,
          "ytick.labelsize": med,
          "figure.titlesize": large}
plt.rcParams.update(params)
plt.style.use("seaborn-whitegrid")
sns.set_style("white")
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv(r"zzz_.csv",encoding="utf-8")
# data = data.drop(["Unnamed: 10","Unnamed: 11"],1)
data

name = ["Age","BMI","Blood pressure","Diastolic blood pressure","Heart rate","HbA1c","GLU","Duration of illness","AGEs fluorescenceintensity","Corneal AGEs concention value"]

data.columns = name 
data

data.corr(method = "pearson")


plt.rcParams["font.sans-serif"] = ["Simhei"]
plt.rcParams["axes.unicode_minus"] = False


plt.figure(figsize = (12,10), dpi = 80)
sns.heatmap(data.corr() #需要输入的相关性矩阵
           ,xticklabels = data.columns  #横坐标标签
           ,yticklabels = data.columns  #纵坐标标签
           ,cmap = "RdYlGn"     #使用的光谱，一般来说都会使用由浅至深，成两头颜色差距较大的光谱
#            ,cmap = "winter"   #不太适合做热力图的光谱
           ,center = 0          #填写数据的中值，注意观察此时的颜色条，大于0的是越来越接近绿色，小于0的越来越靠近红色
#            ,center = 1        #填写数据中的最大值/最小值，则最大值/最小值是最浅或最深的颜色，数据离该极值越远，颜色越浅/颜色越深
           ,annot = True
           )

plt.xticks(fontsize = 12, rotation = 45, horizontalalignment = "right")
plt.yticks(fontsize = 12)
plt.show()

