import  matplotlib.pyplot as plt
# import jdc
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import distance
from scipy.interpolate import make_interp_spline
from tqdm import tqdm # for showing progress bar in for loops

fig = plt.figure(figsize=(20,4))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman' # 全局字体样式
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

ax1 = fig.add_subplot(141)
xlsx_file1 = r".\! plot data\1. Google trends_GDELT_data.xlsx"
Original_data = pd.read_excel(xlsx_file1, sheet_name="All", usecols=[0, 3, 4])
l1,=ax1.plot(Original_data['GDELT_Greek_labels_normalization'], color="#3B8791")
l2,=ax1.plot(Original_data['GoogleTrends_Greek_labels_normalization'], color="#A07936")
ax1.legend(handles=[l1, l2], loc='upper right'
           , labels=['NM', 'CSB']
           , handlelength=0.6 #图例句柄的长度
           , fontsize=15,frameon=False
           ,bbox_to_anchor=(1.03, 0.97))
ax1.set_xticklabels([0,0,50,100,150,200,250],fontsize=15)
ax1.set_yticklabels([0,0,0.2,0.4,0.6,0.8,1.0],fontsize=15)
ax1.set_xlim(0-20, 246+15)
ax1.set_ylim(0 - 0.06, 1 + 0.06)
ax1.set_xlabel('Day (  )',fontsize=18)
ax1.text(136,-0.23,'L', fontstyle='italic',fontsize=18)
ax1.set_ylabel('Normalized Frequency (%)',fontsize=18)


ax2 = fig.add_subplot(142)
xlsx_file2 = r".\! plot data\2. CCM-data(GoogleTrends&GDELT).xlsx"
CCM_data = pd.read_excel(xlsx_file2, usecols=[1, 2, 3])
l1,=plt.plot(CCM_data[["LibSize"]],CCM_data[["GoogleTrends_Greek_labels_normalization:GDELT_Greek_labels_normalization"]]
             , color = '#3B8791')
l2,=plt.plot(CCM_data[["LibSize"]],CCM_data[["GDELT_Greek_labels_normalization:GoogleTrends_Greek_labels_normalization"]]
             , color = '#A07936')
ax2.legend(handles=[l1, l2], loc='lower right'
           , labels=['NM versus CSB', 'CSB versus NM']
           , handlelength=0.6 #图例句柄的长度
           , fontsize=15,frameon=False)
ax2.set_xticklabels([0,0,50,100,150,200,250],fontsize=15)
ax2.set_yticklabels([0,0,0.2,0.4,0.6,0.8,1.0],fontsize=15)
ax2.set_xlim(0-20, 246+15)
ax2.set_ylim(0 - 0.06, 1 + 0.06)
ax2.set_xlabel('Day (  )',fontsize=18)
ax2.text(136,-0.23,'L', fontstyle='italic',fontsize=18)
ax2.set_ylabel('Correlation coefficient (  )',fontsize=18)
ax2.text(-75,0.92,'ρ', fontstyle='italic',rotation=90,fontsize=18)

def shadow_manifold(X, tau, E, L):
    X = X[:L]
    M = {t:[] for t in range((E-1) * tau, L)}
    for t in range((E-1) * tau, L):
        x_lag = []
        for t2 in range(0, E-1 + 1):
            x_lag.append(X[t-t2*tau])
        M[t] = x_lag
    return M
def get_distances(Mx):
    t_vec = [(k, v) for k,v in Mx.items()]
    t_steps = np.array([i[0] for i in t_vec])
    vecs = np.array([i[1] for i in t_vec])
    dists = distance.cdist(vecs, vecs)
    return t_steps, dists
def get_nearest_distances(t, t_steps, dists, E):
    t_ind = np.where(t_steps == t)
    dist_t = dists[t_ind].squeeze()
    nearest_inds = np.argsort(dist_t)[1:E + 1 + 1]
    nearest_timesteps = t_steps[nearest_inds]
    nearest_distances = dist_t[nearest_inds]
    return nearest_timesteps, nearest_distances
class ccm:
    def __init__(self, X, Y, tau=1, E=2, L=500):
        self.X = X
        self.Y = Y
        self.tau = tau
        self.E = E
        self.L = L
        self.My = shadow_manifold(self.Y, self.tau, self.E,self.L)
        self.t_steps, self.dists = get_distances(self.My)
    def causality(self):
        X_true_list = []
        X_hat_list = []
        for t in list(self.My.keys()):
            X_true, X_hat = self.predict(t)
            X_true_list.append(X_true)
            X_hat_list.append(X_hat)
        x, y = X_true_list, X_hat_list
        correl = np.corrcoef(x, y)[0][1]
        return correl
    def predict(self, t):
        eps = 0.000001
        t_ind = np.where(self.t_steps == t)
        dist_t = self.dists[t_ind].squeeze()
        nearest_timesteps, nearest_distances = get_nearest_distances(t, self.t_steps, self.dists, E)
        u = np.exp(
            -nearest_distances / np.max([eps, nearest_distances[0]]))
        w = u / np.sum(u)
        X_true = self.X[t]
        X_cor = np.array(self.X)[nearest_timesteps]
        X_hat = (w * X_cor).sum()
        return X_true, X_hat
    def visualize_cross_mapping(self):
        f, axs = plt.subplots(1, 2, figsize=(12, 6))
        for i, ax in zip((0, 1), axs):
            X_lag, Y_lag = [], []
            for t in range(1, len(self.X)):
                X_lag.append(X[t - tau])
                Y_lag.append(Y[t - tau])
            X_t, Y_t = self.X[1:], self.Y[1:]
            ax.scatter(X_t, X_lag, s=5, label='$M_x$')
            ax.scatter(Y_t, Y_lag, s=5, label='$M_y$', c='y')
            A, B = [(self.Y, self.X), (self.X, self.Y)][i]
            cm_direction = ['Mx to My', 'My to Mx'][i]
            Ma = shadow_manifold(A, tau, E, L)
            Mb = shadow_manifold(B, tau, E, L)
            t_steps_A, dists_A = get_distances(Ma)
            t_steps_B, dists_B = get_distances(Mb)
            timesteps = list(Ma.keys())
            for t in np.random.choice(timesteps, size=3, replace=False):
                Ma_t = Ma[t]
                near_t_A, near_d_A = get_nearest_distances(t, t_steps_A, dists_A, E)
                for i in range(E + 1):
                    A_t = Ma[near_t_A[i]][0]
                    A_lag = Ma[near_t_A[i]][1]
                    ax.scatter(A_t, A_lag, c='b', marker='s')
                    B_t = Mb[near_t_A[i]][0]
                    B_lag = Mb[near_t_A[i]][1]
                    ax.scatter(B_t, B_lag, c='r', marker='*', s=50)
                    ax.plot([A_t, B_t], [A_lag, B_lag], c='r', linestyle=':')
            ax.set_title(f'{cm_direction} cross mapping. time lag, tau = {tau}, E = 3')
            ax.legend(prop={'size': 14})
            ax.set_xlabel('$X_t$, $Y_t$', size=15)
            ax.set_ylabel('$X_{t-1}$, $Y_{t-1}$', size=15)
def plot_ccm_correls(X, Y, tau, E, L,ax1,ax2):
    M = shadow_manifold(Y, tau, E, L)
    t_steps, dists = get_distances(M)
    ccm_XY = ccm(X, Y, tau, E, L)
    ccm_YX = ccm(Y, X, tau, E, L)
    X_My_true, X_My_pred = [], []
    Y_Mx_true, Y_Mx_pred = [], []
    for t in range(tau, L):
        true, pred = ccm_XY.predict(t)
        X_My_true.append(true)
        X_My_pred.append(pred)
        true, pred = ccm_YX.predict(t)
        Y_Mx_true.append(true)
        Y_Mx_pred.append(pred)
    coeff = np.round(np.corrcoef(X_My_true, X_My_pred)[0][1], 2)
    ax2.scatter(X_My_true, X_My_pred, color='#A07936', s=60,marker='+',linewidths=1)
    ax2.set_xlabel('NM( ) (observed)', size=18)
    # axs[1].set_ylabel('$\hat{X}(t)|M_y$ (estimated)', size=15)
    ax2.set_ylabel('CSB( ) (estimated)', size=18)
    # ax2.set_title(f'tau={tau}, E={E}, L={L}, Correlation coeff = {coeff}')
    coeff = np.round(np.corrcoef(Y_Mx_true, Y_Mx_pred)[0][1], 2)
    ax1.scatter(Y_Mx_true, Y_Mx_pred, color='#3B8791', s=60,marker='+',linewidths=1)
    ax1.set_xlabel('CSB( ) (observed)', size=18)
    # axs[0].set_ylabel('$\hat{Y}(t)|M_x$ (estimated)', size=15)
    ax1.set_ylabel('NM( ) (estimated)', size=18)
    # ax1.set_title(f'tau={tau}, E={E}, L={L}, Correlation coeff = {coeff}')


ax3 = fig.add_subplot(143)
ax4 = fig.add_subplot(144)
# A title:tau=1,E=2,L=246,Correlation coeff=0.88
# B title:tau=1,E=2,L=246,Correlation coeff=0.84
X = Original_data['GDELT_Greek_labels_normalization'] # NM
X = np.array(X).tolist()
# print(X)
Y = Original_data['GoogleTrends_Greek_labels_normalization'] #CSB
Y = np.array(Y).tolist()
# print(Y)
L = 246
tau = 1
E = 2
plot_ccm_correls(X, Y, tau, E, L, ax3, ax4)

ax3.set_xticklabels([0,0,0.2,0.4,0.6,0.8,1.0],fontsize=15)
ax3.set_yticklabels([0,0,0.2,0.4,0.6,0.8,1.0],fontsize=15)
ax3.set_xticks([0,0,0.2,0.4,0.6,0.8,1.0])
ax3.set_yticks([0,0,0.2,0.4,0.6,0.8,1.0])
ax3.set_xlim(0 - 0.06, 1 + 0.06)
ax3.set_ylim(0 - 0.06, 1 + 0.06)
ax3.text(0.372,-0.23,'t', fontstyle='italic',fontsize=18)
ax3.text(-0.27,0.37,'t', fontstyle='italic',rotation=90,fontsize=18)

ax4.set_xticklabels([0,0,0.2,0.4,0.6,0.8,1.0],fontsize=15)
ax4.set_yticklabels([0,0,0.2,0.4,0.6,0.8,1.0],fontsize=15)
ax4.set_xticks([0,0,0.2,0.4,0.6,0.8,1.0])
ax4.set_yticks([0,0,0.2,0.4,0.6,0.8,1.0])
ax4.set_xlim(0 - 0.06, 1 + 0.06)
ax4.set_ylim(0 - 0.06, 1 + 0.06)
ax4.text(0.354,-0.23,'t', fontstyle='italic',fontsize=18)
ax4.text(-0.27,0.38,'t', fontstyle='italic',rotation=90,fontsize=18)

plt.savefig('./figure/CCM2_300dpi.jpg', bbox_inches='tight', dpi=300)
plt.show()