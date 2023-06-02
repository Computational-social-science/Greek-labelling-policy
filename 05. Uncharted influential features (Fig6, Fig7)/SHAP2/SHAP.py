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


# force_plot++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
shap.force_plot(shap_values[250], matplotlib=True,show=False,plot_cmap='BrBG')\
        .savefig('./force_plot_300dpi.jpg', bbox_inches='tight', dpi=300)


# heatmap_plot++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
feature_order, values_T_array = shap.plots.heatmap(shap_values)
# clist = ["#F7A065", "#fffea9", "#b7e3a0", "#78c49d", "#44a298","#237f8b","#1D717C"]
clist = ["#44a298","#fffea9","#F7A065","#E1874A","#D7722E","#A44301"]
newcmp = LinearSegmentedColormap.from_list('chaos', clist)

plt.figure(figsize=(16, 4))
plt.rcParams.update({'font.family': 'Times New Roman'})
ax_heatmap = sns.heatmap(values_T_array, cmap=newcmp, linewidths=0, linecolor='gray')

# 设置色条的位置
cbar = ax_heatmap.collections[0].colorbar
cbar.ax.set_position([0.755, 0.11, 0.02, 0.77])  # 根据需要调整数值以确定位置

vmin = values_T_array.min()
vmax = values_T_array.max()

# 设置色条的刻度标签
# cbar.set_ticks([min(vmin,-vmax), max(-vmin,vmax)])  # 根据数据调整数值
cbar.set_ticks([vmin, vmax])
cbar.set_ticklabels(['Low', 'High'])  # 根据需要调整标签
cbar.set_label("SHAP value", size=11, labelpad=-15)

# 设置纵坐标 y轴 刻度标签
yticklabels = ["Greek labels"] + [cols[i] for i in feature_order]
ax_heatmap.set_yticklabels(yticklabels, rotation=0, size=11)
# 获取第一个标签的文本对象
label_text = ax_heatmap.get_yticklabels()[0]
# 设置第一个标签的字体样式为粗体
label_text.set_weight('bold')
# 刷新图像以应用更改
plt.draw()

# 设置横坐标 x轴 的刻度位置
ax_heatmap.set_xticks(np.arange(0, values_T_array.shape[1], 91))
ax_heatmap.set_xticklabels(['31/05/2021', '29/08/2021', '27/11/2021', '25/02/2022', '26/05/2022', '24/08/2022'], rotation=0, size=11)

# 绘制横坐标x轴的黑色坐标轴线
ax_heatmap.axhline(y=values_T_array.shape[0], color='black', linewidth=2)
ax_heatmap.axhline(y=0, color='#363636', linewidth=0.18)

ax_heatmap.set_xlabel('Year', size=13)

# 加格子外框
def highlight_cell(x, y, ax=None, **kwargs):
    rect = plt.Rectangle((x, y), 1, 1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect
for i in range(1, values_T_array.shape[1]):
    for j in range(0, values_T_array.shape[0]+1):
        highlight_cell(i, j, color="#363636", linewidth=0.08)

# 刻度点
ax_heatmap.tick_params(axis="x", bottom=True, length=2)
ax_heatmap.tick_params(axis="y", left=True, length=2)

plt.savefig('./heatmap_plot_300dpi.jpg', bbox_inches='tight', dpi=300)
# plt.savefig('./fig3_300dpi.png', bbox_inches='tight', dpi=300)
# plt.savefig('./fig3_300dpi.tiff', bbox_inches='tight', dpi=300)
# plt.savefig('./fig3_300dpi.svg', bbox_inches='tight', dpi=300)
plt.show()

