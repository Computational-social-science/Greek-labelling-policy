import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
from seaborn import colors
import shap
import matplotlib.ticker as mticker
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import seaborn as sns
plt.style.use('seaborn')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'Times New Roman'
font = {'family': 'serif',
        'weight': "medium"
        }

xlsx_file = "./Greek name_Shap_Date.xlsx"
data = pd.read_excel(xlsx_file, sheet_name="Sheet1", usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
# 选择特征
cols = ['Alpha', 'Beta', 'Gamma', 'Delta','Omicron', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu']
# 训练xgboost回归模型
model = xgb.XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=150)
model.fit(data[cols], data['bigquery_Greek_labels'].values)
# 引用package并且获得解释器explainer
explainer = shap.Explainer(model)
# 获取训练集data各个样本各个特征的SHAP值
shap_values = explainer(data[cols])
# print(shap_values.shape)
# print(data[cols])
# 可以确认基线值就是训练集的目标变量的拟合值的均值。
y_base = explainer.expected_value
# print(y_base)
data['pred'] = model.predict(data[cols])
# print(data['pred'].mean())


# summary_plot++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
clist = ["#1D717C", "#237f8b","#44a298","#fffea9","#F7A065","#E1874A","#D7722E","#A44301"]
newcmp = LinearSegmentedColormap.from_list('chaos', clist)
shap.summary_plot(shap_values,save=True,path='summary_plot',show=True,cmap=newcmp,use_log_scale=True)
