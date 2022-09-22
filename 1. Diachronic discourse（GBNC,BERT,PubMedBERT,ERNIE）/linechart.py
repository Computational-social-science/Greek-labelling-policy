import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
plt.rcParams['font.size'] = 14 # 全局字体大小
plt.rcParams['font.family'] = 'Times New Roman' # 全局字体样式
font = {'family': 'serif',
        'weight': "medium"
        }

data1 = pd.read_excel("./1900-2019（eng_2019）.xlsx", sheet_name='virus', usecols=[14, 15, 16, 17, 18, 19])
data2 = pd.read_excel("./1900-2019（eng_2019）.xlsx",sheet_name='virus species', usecols=[14, 15, 16, 17, 18, 19])
print(data1)
print(data2)


fig = plt.figure(figsize=(20,11))

# 子图间距
plt.subplots_adjust(hspace=0.3)

ax1 = fig.add_subplot(211)
ax1.plot(data1[["year.2"]],data1[["virus.2"]],color='#E27C00', label='virus', linewidth=3)
ax1.plot(data1[["year.2"]],data1[["toxin.2"]],color='#FFA740', label='toxin', linewidth=2)
ax1.plot(data1[["year.2"]],data1[["vira.2"]],color='#E0EC89', label='vira', linewidth=2)
ax1.plot(data1[["year.2"]],data1[["virion.2"]],color='#A6C86D', label='virion', linewidth=2)
ax1.plot(data1[["year.2"]],data1[["venom.2"]],color='#338033', label='venom', linewidth=2)
ax1.legend(loc = 'upper left',frameon=False, fontsize=15)
ax1.set_xlim([1900, 2020])
ax1.set_xlabel('Year')
ax1.set_ylabel('Normalized frequency')
ax1.set_title('a  Diachronic discourse on "virus"', x=0.12, y=1.06, fontweight='bold', fontsize=18)

ax2 = fig.add_subplot(212)
ax2.plot(data2[["year.2"]],data2[["virus species.2"]],color='#E27C00', label='virus species', linewidth=3)
ax2.plot(data2[["year.2"]],data2[["virus taxonomy.2"]],color='#FFA740', label='virus taxonomy', linewidth=2)
ax2.plot(data2[["year.2"]],data2[["virus classification.2"]],color='#E0EC89', label='virus classification', linewidth=2)
ax2.plot(data2[["year.2"]],data2[["virus categories.2"]],color='#A6C86D', label='virus categories', linewidth=2)
ax2.plot(data2[["year.2"]],data2[["virus systematics.2"]],color='#338033', label='virus systematics', linewidth=2)
ax2.legend(loc = 'upper left',frameon=False, fontsize=15)
ax2.set_xlim([1900, 2020])
ax2.set_xlabel('Year')
ax2.set_ylabel('Normalized frequency')
ax2.set_title('b  Diachronic discourse on "virus species"', x=0.147, y=1.06, fontweight='bold', fontsize=18)

plt.savefig('./linechart_300dpi.jpg', bbox_inches='tight', dpi=300)
plt.show()