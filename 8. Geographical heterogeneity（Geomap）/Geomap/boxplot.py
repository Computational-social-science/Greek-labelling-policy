#coding=utf-8
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.gridspec import GridSpec
import mapclassify as mc
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from scipy import interpolate
from plotnine import *
import skmisc
# from skmisc.loess import loess
from scipy.interpolate import make_interp_spline,BSpline


# Smooth function
def np_move_avg(a, n, smooth, mode="same"):
    return np.convolve(a, np.ones((n,)) / n, mode=mode) if smooth else a

#计算中位数
def count_median(lis):
    if len(lis) % 2 == 0:
        mid = float((lis[len(lis) / 2] + lis[len(lis) / 2 - 1])) / 2
    else:
        mid = lis[len(lis) / 2]
    return mid
#计算上下四分位数
def count_quartiles(lis):
    q1 = 1 + (float(len(lis)) - 1) * 1 / 4
    q3 = 1 + (float(len(lis)) - 1) * 3 / 4
    return q1, q3

datafile = 'bokeh-app/data/Geographical name_normalization.csv'
data=pd.read_csv(datafile,usecols=[0,1,2,3],names=['contry','normalization','number','count'])

#low -> high
colorslist = ['#DDEFB1','#FEF4C0','#FEEEBA','#FEE1AA','#FDC68A','#FBBB7F','#FBBA7E','#F5A374','#EF8F6B','#EB7547','#DF5952','#DC402D','#D83428']
mycmaps = colors.LinearSegmentedColormap.from_list('mylist',colorslist,N=100)

fig = plt.figure()

ax1 = plt.subplot(211)
# colorslist = ['#EEEEEE','#3A8791','#033D33']
# mycmaps = colors.LinearSegmentedColormap.from_list('mylist',colorslist,N=60)
# data = gpd.GeoDataFrame(data)
# data1 = np.zeros(l)
# data1 = data1[:,1:len(data1[0])]
# m = int(len(data['number']))
# data1 = np.ones((13,2))
# data1[:,[1]] = np.array([[115]*13]).T
# data1[:,[0]] = np.array([[0,5,10,15,20,25,30,35,40,45,50,55,60]]).T
# print(data['number'])
# extent = [0,60,0,115]
# img = mpimg.imread('color.png')
# img = misc.imresize(img,0.5)
x = data['number'][0:10]
y = data['count'][0:10]
# print(y)
# x_smooth = np.linspace(x.min(),x.max(),300)
# y_smooth = make_interp_spline(x,y)(x_smooth)
# print(y[1])
# for i in range(60):a  `Q1
#     y[i] = np_move_avg(y[i],3,True)
# percentile = np.percentile(data['normalization'][2:193],(25,50,75),interpolation='linear')
# q1 = percentile[0]
# q2 = percentile[1]
# q3 = percentile[2]
# print(q1,q3)
# ax1.axvline(x=q1,ymin=0,color='gray')
# ax1.axvline(x=q2,ymin=0,ymax=0,color='gray')
# ax1.axvline(x=q3,ymin=0,color='gray')
# f = interpolate.interp1d(x,y,kind='linear')
# x_new = np.linspace(np.min(x),np.max(x),5000)
# y_new=f(x_new)
# print(y_new)
# df_interpolate=pd.DataFrame({'x': x_new, 'y':y_new})
# print(df_interpolate)

# 使用scipy.interpolate.spline拟合曲线
x_new = np.linspace(x.min(),x.max(),300) #300 represents number of points to make between T.min and T.max
y_smooth = make_interp_spline(x,y)(x_new)
# print(y_smooth)
# plt.plot(xnew,power_smooth)

# 核密度估计曲线图
# x_data=data['normalization'][2:194]
# df_interpolate=pd.DataFrame({'x': x_data})
# base_plot=(ggplot(aes(x=data['normalization'][2:194]))
#            +geom_density(alpha=1)
#            +scale_fill_hue(s=0.90,l=0.65,h=0.0417,color_space='husl')
#            +theme_light()
#            )
# print(base_plot)

# 正态分布图（数据已使用excel处理）
# Line_plot=(ggplot()
#            +geom_area(df_interpolate,aes('x','y'),size=1,fill='none')
#            +geom_line(df_interpolate,aes('x','y'),size=1,color='gray')
#            +scale_fill_cmap(name=mycmaps))
# print(Line_plot)
# 正态分布曲线（平滑）
# l = loess(x,y)
# l.fit()
# pred = l.predict(x,stderror=True)
# y_fit = pred.values
# ax1.plot(x,y_fit,color='gray',linewidth=1)
# plot_loess=(ggplot())

# fill_colors = matplotlib.cm.get_cmap(mycmaps)
# col = fill_colors(np.linspace(0,1,301))
# print(col[0])
# print(zip(x_new,y_smooth))
# for i,(xx,yy) in enumerate(zip(x_new,y_smooth)):
#     # print('{}:({},{})'.format(col[i],xx,yy))
#     ax1.fill_between(xx,yy,y2=0,color=col[i])
# print('{}:({},{})'.format(col[1],x_new[1],y_smooth[1]))
# print(type(mycmaps))
# fill_colors = colors.Colormap(colorslist,100)
# # ax1.fill_between(x_new,y_smooth,y2=0,cmap=mycmaps)
# ax1.set(xlim=(0,max(x_new)), ylim=(-0.003,max(y_smooth)), autoscale_on=False)
# a = np.array([[1,2,3],
#               [1,2,3]])
# ax1.imshow(a, interpolation='bicubic', extent=(0,max(x_new), -0.003,max(y_smooth)),cmap=mycmaps,aspect='auto')
# ax1.fill_between(x_new,y_smooth,max(y_smooth),color='#EFF5FA')
# ax1.plot(x_new,y_smooth,color='gray',linestyle='-',linewidth=2)
# # ax1.imshow(img,aspect='auto',extent=[0,59,0,60])
# # ax1.xticks([q1,q2,q3],
# #            [q1,q2,q3])
# # ax1.tick_params(axis='x',width=0)
# ax1.axis('off')  # 去坐标轴

ax2 = plt.subplot(212,facecolor='#EFF5FA')
print(data['normalization'][5:151])
f = ax2.boxplot(
    # 绘图数据
    x = data['normalization'][5:151],
    vert = False,
    widths=0.2,
    boxprops = {'color':'gray'},
    showcaps=False,
    flierprops = {'marker':'o','markerfacecolor':'gray','color':'none'},
    medianprops= {'linestyle':'-','color':'gray'}
    )
# background_colors = ['#EFF5FA']
# mycmaps = colors.LinearSegmentedColormap.from_list('mylist',background_colors)
# ax2.imshow(data['normalization'][2:193],cmap=mycmaps,extent=[2,193,0,60],aspect='auto')
ax2.axis('off')  # 去坐标轴
plt.show()