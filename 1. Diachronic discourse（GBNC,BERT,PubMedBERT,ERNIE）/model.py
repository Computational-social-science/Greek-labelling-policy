from transformers import *
import torch
import numpy as np
import xlwt
import pandas as pd
import random
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
import numpy as NP
import warnings
import matplotlib as mpl
from pylab import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import os
from mpl_toolkits.axes_grid1 import ImageGrid
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model
from matplotlib.gridspec import GridSpec

'''
运行关掉VPN！！！
'''

# 获取词向量
def get_embedding(model,tokenizer,words):
    embeddings = []
    for word in words:
        encoded_input=tokenizer(word,return_tensors='pt')
        output=model(**encoded_input)
        embeddings.append(torch.mean(output.last_hidden_state[0, :, :], dim=0))
    return embeddings
# 余弦相似度
def calculate_similariy(embedding1, embedding2):
    return (torch.dot(embedding1, embedding2) / (torch.norm(embedding1) * torch.norm(embedding2))).item()
# 求相似矩阵
def get_score(words1,words2,embeddings1,embeddings2):
    score = np.zeros(shape=(len(words1), len(words2)))
    for i in range(len(words2)):
        for j in range(len(words1)):
            score[i, j] = calculate_similariy(embeddings2[i], embeddings1[j])
    return score
# 使用模型求词组间的相似性
def getSimilarity(txt,words1,words2):
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(txt)
    model = AutoModel.from_pretrained(txt)
    # 获取对应模型的词向量
    embeddings1 = get_embedding(model,tokenizer,words1)
    embeddings2 = get_embedding(model,tokenizer,words2)
    # 获取矩阵相似系数
    score = get_score(words1,words2,embeddings1,embeddings2)
    return score



# 目标词
words1 = ["virus","virion","venom","vira","toxin"]
words2 = ["virus species","virus categories","virus classification","virus taxonomy","virus  systematics"]
titles = ["c  BERT","d  PubMedBERT","e  ERNIE"]
# BERT
score_BERT = getSimilarity('bert-base-uncased',words1,words2)
print(score_BERT)
# PubMedBERT
score_PubMedBERT = getSimilarity('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',words1,words2)
print(score_PubMedBERT)
# ERNIE
score_ERNIE = getSimilarity('nghuyong/ernie-2.0-large-en',words1,words2)
print(score_ERNIE)




# 画图
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
plt.rcParams['font.family'] = 'Times New Roman' # 全局字体样式

fig = plt.figure(figsize=(20,5.5))
xLabel = [""]+words1
yLabel = [""]+words2
scores = [score_BERT,score_ERNIE,score_PubMedBERT]
# plt.subplots_adjust(wspace=0.3,hspace=0.3)

grid = ImageGrid(fig, 111,
                          nrows_ncols=(1,3), # 表示创建一个1行3列的画布
                          axes_pad=0.15,
                          share_all=True, # 表示所画的图像公用x坐标轴和y坐标轴
                          cbar_location="right", # 表示colorbar位于图像的右侧
                          cbar_mode="single", # 表示三个图像公用一个colorbar
                          cbar_size="5.5%", # 表示colorbar的尺寸，默认值为5%
                          cbar_pad=0.2, # 表示图像与colorbar之间的填充间距，默认值为5%
                         )
# for ax in grid:
#     im = ax.imshow(np.random.random((10,10)), vmin=0, vmax=1)

for ax,i in zip(grid,range(len(scores))): # 0 1 2
    # ax = fig.add_subplot(131+i)
    # 定义横纵坐标的刻度
    ax.set_yticks(np.arange(scores[i].shape[1]+1)-.5,minor=True)
    ax.set_yticklabels(yLabel,fontsize=14, rotation=360)
    ax.set_xticks(np.arange(scores[i].shape[0]+1)-.5,minor=True)
    ax.set_xticklabels(xLabel,fontsize=14, rotation=0)
    # 设置边框主刻度线，颜色为白色，线条格式为'-',线的宽度为3
    ax.grid(which="minor",color="w", linestyle='-', linewidth=3)
    # spines是连接轴刻度标记的线，而且标明了数据区域的边界
    for edge, spine in ax.spines.items():
        # spine.set_visible(False)
        spine.set_linewidth('7.5')
        spine.set_color('w')
    clist=['#338033','#A6C86D','#E0EC89','#ECF3B7','#FDF7A9','#FFC580','#FFA740','#E27C00'] #自定义图表色系
    newcmp = LinearSegmentedColormap.from_list('chaos',clist)
    cmap1 = cm.get_cmap(newcmp, 10) # jet doesn't have white color
    cmap1.set_bad('w') # default value is 'k'
    #作图并选择热图的颜色填充风格，这里选择自定义
    im = ax.imshow(scores[i], interpolation="nearest", cmap=cmap1,vmin=0.4,vmax=1)
    # 为每一个格子加上数值
    for x in range(0,len(scores[i])):
        for y in range(0,len(scores[i][x])):
            ax.text(x, y+0.03, '%.3f' % scores[i][y][x],
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=14,
                    )
    # 增加标题
    if i==0:
        ax.set_title(titles[i], fontsize=18, x=0.12, y=1.02, fontweight='bold')
    elif i==1:
        ax.set_title(titles[i], fontsize=18, x=0.25, y=1.02, fontweight='bold')
    else:
        ax.set_title(titles[i], fontsize=18, x=0.16, y=1.02, fontweight='bold')


ax.cax.colorbar(im)
ax.cax.toggle_label(True)



plt.savefig('./xBERT_300dpi.jpg', bbox_inches='tight', dpi=300)
# show
plt.show()



