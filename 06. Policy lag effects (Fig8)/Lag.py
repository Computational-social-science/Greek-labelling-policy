import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

Geographical = ['UK variant', 'South Africa variant', 'Brazil variant', 'India variant', 'California variant'
    , 'Nigeria variant', 'Philippines variant', 'US variant']
Greek = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Eta', 'Theta', 'Iota']
colors = ['#D08234', '#4BB9D0', '#009F85']

fig =plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)
ax.xaxis.set_major_locator(MultipleLocator(80)) # 0-80为主刻度线
ax.xaxis.set_minor_locator(MultipleLocator(20)) # 0-80中又每10个为单位生成次刻度线


# Recognition lag
df = pd.read_excel(r'D:\!cy\! ZJSU\1 ZJSU 科研\2021.9.11 Greek name\2. Lag 时间滞后性\Lag.xlsx',sheet_name='Recognition lag')
i = 21.7
j = 0
while(i>=0):
    plt.scatter(df[Geographical[j]].first_valid_index(), i, s=40, color=colors[0], marker = 'o')
    plt.hlines(i, df[Geographical[j]].first_valid_index(), df[Geographical[j]].last_valid_index(), color=colors[0]
               , label='',linewidth=1.8)
    plt.scatter(df[Geographical[j]].last_valid_index(), i, s=40, color=colors[0], marker = '^')
    i -= 2.8
    j += 1

# Response lag
df = pd.read_excel(r'D:\!cy\! ZJSU\1 ZJSU 科研\2021.9.11 Greek name\2. Lag 时间滞后性\Lag.xlsx',sheet_name='Response lag')
i = 21
j = 0
TNR_font = {"family" : "Times New Roman"}
while(i>=0):
    plt.scatter(df[Geographical[j]].first_valid_index(), i, s=40, color=colors[1], marker = '^')
    plt.hlines(i, df[Geographical[j]].first_valid_index(), df[Geographical[j]].last_valid_index(), color=colors[1],
               label='',linewidth=1.8)
    plt.scatter(df[Geographical[j]].last_valid_index(), i, s=40, color=colors[1], marker = 's')
    plt.text(-20, i + 0.3, Greek[j], fontsize=10, fontdict=TNR_font, ha='right')
    plt.text(-20, i - 0.3, '(' + Geographical[j] + ')', fontsize=10, fontdict = TNR_font, ha='right')
    i -= 2.8
    j += 1


# Transmission lag
df = pd.read_excel(r'D:\!cy\! ZJSU\1 ZJSU 科研\2021.9.11 Greek name\2. Lag 时间滞后性\Lag.xlsx',sheet_name='Transmission lag')
i = 20.3
j = 0
while(i>=0):
    plt.scatter(df[Geographical[j]].first_valid_index(), i, s=40, color=colors[2], marker = 's')
    plt.hlines(i, df[Geographical[j]].first_valid_index(), df[Geographical[j]].last_valid_index(), color=colors[2],
               label='',linewidth=1.8)
    plt.scatter(df[Geographical[j]].last_valid_index(), i, s=50, color=colors[2], marker = 'p')
    i -= 2.8
    j += 1

# for i in range(4,29,4):
#     plt.hlines(i, -50, 815, color="gray", label='', linewidth=0.5, linestyles=':') #横线分隔

for i in range(0,801,20):
    plt.vlines(i,0,22,colors='gray',alpha = 0.1)

# ax.grid(True)
# ax.grid(axis="y")

plt.ylim([0,22])
plt.yticks([])
plt.xlim([0 - 10,800 + 15])
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2) #设置边框粗细
plt.savefig('./lag_300dpi_原.jpg', bbox_inches='tight', dpi=300)
plt.show()
