 # 加载模块
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
from seaborn import colors

plt.style.use('seaborn')
import shap
import matplotlib.ticker as mticker
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.ticker as ticker


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.size'] = 8

plt.rcParams['font.family'] = 'Times New Roman'
font = {'family': 'serif',
        'weight': "medium"
        }

# 读取数据，目标变量Geographical_descroptors:Greek_labels是Greek_labels在CCM模型中对Geographical_descroptors的相关系数
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


# Fig1
# shap.summary_plot(shap_values,save=True,path='./figure/shap1/summary_plot',show=True,cmap='BrBG')


# Fig2
# shap.force_plot(shap_values[250], matplotlib=True,show=False,plot_cmap='BrBG')\
#         .savefig('./figure/shap2/force_plot_300dpi.jpg', bbox_inches='tight', dpi=300) #plot_cmap='RdBu'


# Fig3
# fig=plt.gcf()
# shap.plots.heatmap(shap_values)
# fig.savefig('./figure/shap3/heatmap_300dpi.jpg', bbox_inches='tight', dpi=300)


# Fig4
shap_pca = PCA(n_components=2).fit_transform(shap_values.values)
def embedding_plot(plt,embedding, values, label, list_ticks=None):
    # plt.figure(figsize=(5,5))
    plt.scatter(embedding[:,0],
               embedding[:,1],
               c=values,
               linewidth=0, alpha=1.0,s=18,
               # cmap=shap.plots.colors.red_blue
               cmap = 'BrBG'
                )
    cb = plt.colorbar(aspect=40, orientation="horizontal")
    tick_locator = ticker.MaxNLocator(nbins=3)  # colorbar上的刻度值个数
    cb.locator = tick_locator
    cb.set_ticks(list_ticks) # 自定义刻度
    cb.update_ticks()
    cb.set_alpha(1)
    cb.draw_all()
    cb.outline.set_linewidth(0)
    cb.ax.tick_params('x', length=0,labelsize=8) #刻度字体
    cb.set_label(label,fontsize=8)  # 标签字体
    cb.ax.xaxis.set_label_position('top')
    plt.gca().axis("off")
    # if show:
    #     plt.show()
plt.figure(figsize=(10,3))
plt.subplot(1,4,1)
embedding_plot(plt,shap_pca, shap_values.values.sum(1), "Greek labels",[0,20000,40000])
# plt.text(-9100,3500,"B",ha = 'left',fontsize=12,FontWeight='bold')
plt.text(23000,-900,"=",ha = 'right',fontsize=24)
plt.subplots_adjust(left=None, bottom=None, right=1, top=None, wspace=None, hspace=None)
plt.subplot(1,4,2)
embedding_plot(plt,shap_pca, shap_values.values[:,3], "Delta",[0,6000,12000]) # x[:,n]就是取所有集合的第n个数据
# plt.text(-9100,3500,"C",ha = 'left',fontsize=12,FontWeight='bold')
plt.text(23000,-900,"+",ha = 'right',fontsize=24)
plt.subplot(1,4,3)
embedding_plot(plt,shap_pca, shap_values.values[:,10], "Omicron",[-600,0,800])
# plt.text(-9100,3500,"D",ha = 'left',fontsize=12,FontWeight='bold')
plt.text(23000,-900,"+",ha = 'right',fontsize=24)
plt.subplot(1,4,4)
plt.text(0.5,0.6,"10",ha = 'center',fontsize=13)
plt.text(0.5,0.5,"other labels",ha = 'center',fontsize=13)
plt.gca().axis("off")
plt.savefig('./figure/shap4/embedding_plot_300dpi.jpg', bbox_inches='tight', dpi=300)
plt.show()