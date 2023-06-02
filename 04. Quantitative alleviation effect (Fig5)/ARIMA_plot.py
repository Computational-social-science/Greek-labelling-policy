import pandas as pd
import numpy as np
import statsmodels #时间序列
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy import  stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_pacf    #偏自相关图
from statsmodels.graphics.tsaplots import plot_acf  #自相关图
from statsmodels.tsa.stattools import adfuller #ADF检验
import itertools
from statsmodels.tsa.arima.model import ARIMA #ARIMA模型
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

sheet_names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Eta", "Theta", "Iota", "Lambda", "Mu", "All"]
sheet_name = sheet_names[10]
ARIMA1 = [0,4,3]
ARIMA2 = [6,2,2]
variable_name1 = "first_stage"
variable_name2 = "Greek_accumulate"


# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False
# plt.rcParams['font.size'] = 13
# plt.rcParams['font.family'] = 'Times New Roman'
TNR_font = {"family" : "Times New Roman"}

'''
1.数据预处理
'''
def pretreatment(inputfile, sheet_name):
    data = pd.read_excel(inputfile ,sheet_name= sheet_name)
    data_1 = data[variable_name1] # first stage
    data_2 = data[variable_name2] # Greek label accumulate
    # print(len(data_1),len(data_2))
    return data_1,data_2, data
data_1, data_2, data = pretreatment(r'.\data.xlsx', sheet_name)


'''
2.模型构建并进行可视化
'''
# Geographical descroptors - first stage
model = ARIMA(data_1[0:168],order = (ARIMA1[0],ARIMA1[1],ARIMA1[2])).fit() # ARIMA  p,d,q
forecast_data = model.forecast(30) # 预测30天数据
fit_data = pd.Series(model.fittedvalues, copy=True) # 拟合数据
pred_conf_50 = model.get_forecast(steps=30).conf_int(alpha=0.5)# 置信区间数据，有50%的可能数据会落在该区域
pred_conf_60 = model.get_forecast(steps=30).conf_int(alpha=0.4)
pred_conf_70 = model.get_forecast(steps=30).conf_int(alpha=0.3)
pred_conf_80 = model.get_forecast(steps=30).conf_int(alpha=0.2)
pred_conf_90 = model.get_forecast(steps=30).conf_int(alpha=0.1)
fig = plt.figure(figsize=(8, 5), dpi=100)
ax1 = fig.add_subplot(111)
ax1.fill_between(pred_conf_90.index-1, pred_conf_90.iloc[:,0]*0.975, pred_conf_90.iloc[:,1]/0.975,color='#F8F5EC')
ax1.fill_between(pred_conf_80.index-1, pred_conf_80.iloc[:,0]*0.975, pred_conf_80.iloc[:,1]/0.975,color='#F5F5EC')
ax1.fill_between(pred_conf_70.index-1, pred_conf_70.iloc[:,0]*0.975, pred_conf_70.iloc[:,1]/0.975,color='#F3EDDD')
ax1.fill_between(pred_conf_60.index-1, pred_conf_60.iloc[:,0]*0.975, pred_conf_60.iloc[:,1]/0.975,color='#F1EADB')
ax1.fill_between(pred_conf_50.index-1, pred_conf_50.iloc[:,0]*0.975, pred_conf_50.iloc[:,1]/0.975,color='#EDE3CF')
ax1.fill_between(fit_data.index, fit_data*0.97, fit_data/0.97, facecolor='#EDE3CF')
# ax1.plot(data["variant_accumulate"][0:168], '#B68C42', label='real1', linewidth=1)
# ax1.plot(data["variant_accumulate"][167:], '#B68C42', label='real2', linewidth=1)
# ax1.scatter(data["variant_accumulate"].index, data["variant_accumulate"], label='actual Geographical descriptors'
#             , color = '#B68C42', s = 2)
ax1.plot(data["variant_accumulate"], '#B68C42', label='Geographical descriptors (actual)', linewidth=2)
ax1.plot(forecast_data.index-1, forecast_data,color='#B68C42', label='Geographical descriptors (forecast)', linewidth=2, linestyle='--')
# ax1.set_yticks([0, 20, 40, 60, 80, 100, 120]) #左边是Geographical descroptors
# ax1.set_yticklabels([0, 20, 40, 60, 80, 100, 120], fontsize=9, font = TNR_font)
ax1.set_ylim(0 - 6,126.3)
ax1.set_xlim(0-30,490+30)
# ax1.set_xlabel("Date", weight = 'bold')
ax1.set_xlabel("   ", weight = 'bold')
ax1.set_ylabel("Geographical descriptors (accumulated value)", fontweight = 'bold', labelpad = 3, fontsize = 10.5)
# ax1.set_xticklabels([0, 168, 424], fontsize=9, font = TNR_font) #x轴字体设置
ax1.xaxis.set_major_locator(MultipleLocator(64))
ax1.xaxis.set_minor_locator(MultipleLocator(4))

# 5.31的虚线
plt.vlines(168, -5, 130, 'gray', '--', label='', linewidth=0.7) # 垂直的线

# Greek name
model = ARIMA(data_2[169:],order = (ARIMA2[0],ARIMA2[1],ARIMA2[2])).fit() # ARIMA  p,d,q
forecast_data = model.forecast(30) # 预测30天数据
fit_data = pd.Series(model.fittedvalues, copy=True) # 拟合数据
pred_conf_50 = model.get_forecast(steps=30).conf_int(alpha=0.5) # 置信区间数据，有50%的可能数据会落在该区域
pred_conf_60 = model.get_forecast(steps=30).conf_int(alpha=0.4)
pred_conf_70 = model.get_forecast(steps=30).conf_int(alpha=0.3)
pred_conf_80 = model.get_forecast(steps=30).conf_int(alpha=0.2)
pred_conf_90 = model.get_forecast(steps=30).conf_int(alpha=0.1)
ax2 = ax1.twinx() #共享xz轴，设置双y轴
ax1_yticks = ax1.get_yticks() # get positions of the ax1 y-ticks in data coordinates
ax2.fill_between(pred_conf_90.index-1, pred_conf_90.iloc[:,0]*0.975, pred_conf_90.iloc[:,1]/0.975,color='#EBF2F5')
ax2.fill_between(pred_conf_80.index-1, pred_conf_80.iloc[:,0]*0.975, pred_conf_80.iloc[:,1]/0.975,color='#DFF0F1')
ax2.fill_between(pred_conf_70.index-1, pred_conf_70.iloc[:,0]*0.975, pred_conf_70.iloc[:,1]/0.975,color='#D6EAEE')
ax2.fill_between(pred_conf_60.index-1, pred_conf_60.iloc[:,0]*0.975, pred_conf_60.iloc[:,1]/0.975,color='#D0E9EC')
ax2.fill_between(pred_conf_50.index-1, pred_conf_50.iloc[:,0]*0.975, pred_conf_50.iloc[:,1]/0.975,color='#C7E2E7')
ax2.fill_between(fit_data.index, fit_data*0.97, fit_data/0.97, facecolor='#C7E2E7')
# ax2.scatter(data["Greek_accumulate"].index, data["Greek_accumulate"], label='actual Greek labels'
#             , color = '#7ABFC9', s = 2)
ax2.plot(data["Greek_accumulate"], '#7ABFC9', label='Greek labels (actual)', linewidth=2)
ax2.plot(forecast_data.index-1, forecast_data,color='#7ABFC9', label='Greek labels (forecast)', linewidth=2, linestyle='--')
# ax2.set_yticks([0, 120, 240, 360, 480, 600]) #右边是Greek name
# ax2.set_yticklabels([0, 100, 200, 300, 400, 500, 600], fontsize=9, font = TNR_font)
ax2.set_ylabel("Greek labels (accumulated value)", fontweight = 'bold', labelpad = 8, fontsize = 10.5)

#去掉边框
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_linewidth(2) #设置边框粗细

fig.set_tight_layout(True)
fig.legend(loc = 'upper left',bbox_to_anchor=(0.085, 0.97),frameon=False, fontsize=8, prop=TNR_font
           ,handlelength=1 #图例句柄的长度
            )
# plt.xticks([0, 168, 424],['14/12/2020', '31/05/2021', '10/02/2022'], fontsize=9)
# plt.xticks(np.arange(0,424,4))
plt.savefig('./figures/ALL/300dpi.jpg', bbox_inches='tight', dpi=300)
plt.show()
