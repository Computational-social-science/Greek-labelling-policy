'''
    通过信息准则定阶（AIC、BIC、HQIC）来确定模型的 p,d,q
    并使用DW检验值（当D-W检验值接近于2时，不存在自相关性，说明模型较好）
    和QQ图（通过qq图可以看出，残差基本满足了正态分布）对拟合模型效果进行检验
'''
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
from statsmodels.stats.diagnostic import acorr_ljungbox #白噪声检验
from statsmodels.stats.stattools import durbin_watson #DW检验
from statsmodels.graphics.api import qqplot #qq图

sheet_names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Eta", "Theta", "Iota", "Lambda", "Mu", "All"]
sheet_name = sheet_names[10]
variable_name1 = "first_stage"
variable_name2 = "Greek_accumulate"


'''
1.数据预处理
'''
def pretreatment(inputfile, sheet_name):
    data = pd.read_excel(inputfile ,sheet_name= sheet_name)
    data_1 = data[variable_name1] # first stage
    data_2 = data[variable_name2] # Greek label accumulate
    # print(len(data_1),len(data_2))
    return data_1,data_2
data_1, data_2 = pretreatment(r'.\data.xlsx', sheet_name)


'''
2.时间序列的差分d——将序列平稳化
'''
def Diff(data_1,data_2):
    # first stage
    # fig1 = plt.figure(figsize=(8, 8))
    # ax1 = fig1.add_subplot(311)
    # ax1.set_title("Geographical descroptors first stage")
    # data_1.plot(ax=ax1)
    # ax2 = fig1.add_subplot(312)
    diff1_1 = data_1.diff(1) # 一阶差分
    # diff1_1.plot(ax=ax2)
    # ax3 = fig1.add_subplot(313)
    diff1_2 = data_1.diff(1).diff(1) # 二阶差分
    # diff1_2.plot(ax=ax3)
    # # Greek label accumulate
    # fig2 = plt.figure(figsize=(8, 8))
    # ax4 = fig2.add_subplot(311)
    # ax4.set_title("Greek label accumulate")
    # data_2.plot(ax=ax4)
    # ax5 = fig2.add_subplot(312)
    diff2_1 = data_2.diff(1)  # 一阶差分
    # diff2_1.plot(ax=ax5)
    # ax6 = fig2.add_subplot(313)
    diff2_2 = data_2.diff(1).diff(1)  # 二阶差分
    # diff2_2.plot(ax=ax6)
    # plt.show()
    return diff1_2,diff2_2
diff1_2, diff2_2 = Diff(data_1,data_2)
def ADF_test(timeseries): ## 用于检测序列是否平稳
    x = np.array(timeseries)
    adftest = adfuller(x, autolag='AIC')
    #print (adftest)
    if adftest[0] < adftest[4]["1%"] and adftest[1] < 10**(-8):
    # 对比Adf结果和10%的时的假设检验 以及 P-value是否非常接近0(越小越好)
        print("差分后序列平稳")
        return True
    else:
        print("非平稳序列")
        return False
diff1 = diff1_2 # 经过差分最终确定的平稳数据
d1 = 2
while(True):
    if(ADF_test(diff1.dropna())==True):
        break
    else:
        diff1 = diff1.diff(1)
        d1 += 1
diff2 = diff2_2
d2 = 2
while(True):
    if(ADF_test(diff2.dropna())==True):
        break
    else:
        diff2 = diff2.diff(1)
        d2 += 1
print("Geographical descriptors first stage: d=",d1)
# ADF_test(diff1.dropna())
print("Greek label accumulate：d=",d2)
# ADF_test(diff2.dropna())


'''
3.建立模型——参数选择
定阶方法主要为两种：
（1）ACF和PACF 利用拖尾和截尾来确定
（2）信息准则定阶（AIC、BIC、HQIC）
'''

'''
3.1.分别画出ACF(自相关)和PACF（偏自相关）图像
'''
# fig2 = plt.figure(figsize=(8, 8))
# ax4 = fig2.add_subplot(2,1,1)
# plot_acf(diff2.dropna(),lags=20,ax=ax4) #延迟数
# ax5 = fig2.add_subplot(2,1,2)
# plot_pacf(diff2.dropna(),lags=20,ax=ax5,method='ywm')
# plt.show()

'''
3.2.信息准则定价
'''
def detetminante_order(timeseries): #信息准则定阶：AIC、BIC、HQIC，定阶要使用差分后平稳的数据
    #AIC
    AIC = sm.tsa.arma_order_select_ic(timeseries,\
        max_ar=6,max_ma=6,ic='aic')['aic_min_order']
    #BIC
    BIC = sm.tsa.arma_order_select_ic(timeseries,max_ar=6,\
           max_ma=6,ic='bic')['bic_min_order']
    #HQIC
    HQIC = sm.tsa.arma_order_select_ic(timeseries,max_ar=6,\
                 max_ma=6,ic='hqic')['hqic_min_order']
    print('the AIC is{}\nthe BIC is{}\nthe HQIC is{}'.format(AIC,BIC,HQIC))
def heatmap_AIC(timeseries):
    # 设置遍历循环的初始条件，以热力图的形式展示，原理同AIC，BIC，HQIC定阶
    p_min = 0
    q_min = 0
    p_max = 5
    q_max = 5
    d_min = 0
    d_max = 5
    # 创建Dataframe,以AIC准则
    results_aic = pd.DataFrame(index=['AR{}'.format(i) \
                                      for i in range(p_min, p_max + 1)], \
                               columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])
    # itertools.product 返回p,q中的元素的笛卡尔积的元组
    for p, d, q in itertools.product(range(p_min, p_max + 1), \
                                     range(d_min, d_max + 1), range(q_min, q_max + 1)):
        if p == 0 and q == 0:
            results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
            continue
        try:
            model = sm.tsa.ARIMA(timeseries, order=(p, d, q))
            results = model.fit()
            # 返回不同pq下的model的AIC值
            results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.aic
        except:
            continue
    results_aic = results_aic[results_aic.columns].astype(float)
    # print(results_bic)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(results_aic,
                     # mask=results_aic.isnull(),
                     ax=ax,
                     annot=True,  # 将数字显示在热力图上
                     fmt='.2f',
                     )
    ax.set_title('AIC')
    # plt.show()
'''
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
# print("Geographical descriptors first stage: ")
# detetminante_order(diff1.dropna())
# print("Greek label accumulate：")
# detetminante_order(diff2.dropna())
# plt.show()
'''
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

'''
4.模型构建并进行检验
'''
def random_test(timeseries) : #随机性检验（白噪声检验）
    p_value = acorr_ljungbox(timeseries, lags=1)  # p_value 返回二维数组，第二维为P值
    if p_value[1] < 0.05:
        print("非随机性序列")
        return  True
    else:
        print("随机性序列,即白噪声序列")
        return False
def evaluate_model(model):
    ###（1）利用QQ图检验残差是否满足正态分布
    resid = model.resid  # 求解模型残差
    # print(resid.values)
    plt.figure(figsize=(12, 8))
    qqplot(resid, line='q', fit=False)
    ###（2）利用D-W检验,检验残差的自相关性
    print('D-W检验值为{}'.format(durbin_watson(resid.values)))
'''
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
# model1 = ARIMA(data_1[0:168],order = (1,2,1)).fit() # ARIMA  p,d,q
# model2 = ARIMA(data_2[169:],order = (2,2,1)).fit() # ARIMA  p,d,q
# evaluate_model(model1)
# evaluate_model(model2)
# plt.show()
'''
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''



