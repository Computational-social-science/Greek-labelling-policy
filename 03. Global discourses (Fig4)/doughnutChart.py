import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams['font.sans-serif'] = ['SimHei']
# 配置坐标轴刻度值模式，显示负号
mpl.rcParams['axes.unicode_minus'] = False

# 定义数据
weight1 = [84.99, 8.95, 3.40, 2.66] #外层
weight2 = [42.8, 30.47, 15.86, 10.87] #内层，Nature
cs1 = ['#3B8791', '#4DAAB7', '#7ABFC9', '#B2DBE0'] #外层
cs2 = ['#A07936', '#B68C42', '#C9A769', '#E0CCA8'] #内层，Nature

# 对数据进行排序
# x = list(zip(elements, weight1, weight2, cs))
# x.sort(key=lambda e: e[1], reverse=True)
# [elements, weight1, weight2, cs] = list(zip(*x))
# print(x)

# 初始化图表区
fig = plt.figure(figsize=(6, 6))

# 绘制外层圆环
wedges1, texts1, autotexts1 = plt.pie(x=weight1,
                                      autopct='%3.1f%%',
                                      radius=1,
                                      pctdistance=1.1, #标签所处的位置
                                      startangle=90,
                                      counterclock=False,
                                      colors=cs1,
                                      # 锲形块边界属性字典
                                      wedgeprops={'edgecolor': 'white',
                                                  'linewidth': 0,
                                                  'linestyle': '-'
                                                  },
                                      # 锲形块标签文本和数据标注文本的字体属性
                                      textprops=dict(color='white',  #  字体颜色
                                                     fontsize=10,
                                                     family='Arial'
                                                     )
                                     )

# 绘制内层圆环
wedges2, texts2, autotexts2 = plt.pie(x=weight2,
                                      autopct='%3.1f%%',
                                      radius=0.7,
                                      pctdistance=0.15,
                                      startangle=90,
                                      counterclock=False,
                                      colors=cs2,
                                      # 锲形块边界属性字典
                                      wedgeprops={'edgecolor': 'white',
                                                  'linewidth': 0,
                                                  'linestyle': '-'
                                                  },
                                      # 锲形块标签文本和数据标注文本的字体属性
                                      textprops=dict(color='white',  #  字体颜色
                                                     fontsize=10,
                                                     family='Arial'
                                                     )
                                     )

circle1 = plt.Circle((0,0),0.7, color = 'white', fill = False, clip_on = False, linewidth=3.5)
ax=plt.gca()
ax.add_artist(circle1)
plt.axis("equal")

# 绘制中心空白区域
plt.pie(x=[1],
        radius=0.4,
        colors=[fig.get_facecolor()]
       )

plt.savefig('./doughnutChart_300dpi.jpg', bbox_inches='tight', dpi=300)
plt.show()
