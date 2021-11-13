######## Visualization of Cosine Similarities (Statistical) #########
#####################################################################
import matplotlib
from matplotlib import pyplot as plt
print(matplotlib.rcParams['text.usetex'])
import seaborn as sns
from scipy import stats
import pandas as pd
import statsmodels.api as sm
from utils.utils import project_away_pc
import os
import numpy as np
import torch
import clip
from utils.utils import l2norm
from utils.global_dir_utils import create_dt
device =torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')



texts = [
    "Grumpy",  "Asian",  "Female", "Smiling", "Lipstick", "Eyeglasses", \
    "friendly", "Bangs", "Black hair", "Blond hair", "Straight hair", "Bushy eyebrows", "Chubby", "Earrings", "Goatee", "Receding hairline", \
    "Grey hair", "Brown hair", "Wavy hair", "Wear suit", "Double chin", "Bags under eyes","Big nose", "Big lips",  "Old",\
     "Arched eyebrows","Muslim", "Tanned", "Pale", "Fearful", "He is feeling pressed","irresponsible", "inefficient", "intelligent", "Terrorist", "homocide", "handsome"
    ]
neutrals = [""]*len(texts)

s_dict = np.load(os.path.join("npy", "ffhq", "fs3.npy"))
s_dict_istr = project_away_pc(s_dict) # 6048, 512
s_dict_center = s_dict - s_dict.mean(0, keepdims=True)
x = np.vstack([s_dict, s_dict_center, s_dict_istr])

model, preprocess = clip.load('ViT-B/32', device=device)
dts=[]
t = 4
for i, text in enumerate(texts[:t]):
    dt = create_dt(text, model=model, neutral=neutrals[i])
    dts.append(dt)
dts = torch.stack(dts, dim=0).squeeze(dim=1)
def create_df(x, dts, texts):
    anchors = torch.Tensor(x).to(device)
    cos_sim =  anchors @ dts.T # 6048*2, 8
    x = (cos_sim - cos_sim.mean())/cos_sim.std()
    x = x.detach().cpu().numpy()
    x1, x2, x3 = x[:6048, :], x[6048:12096, :], x[12096:, :] # 6048, t

    fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(5, 5))
    ax1.boxplot(x1, notch=False, showmeans=False, showfliers=False)
    ax1.set_xticks([])
    ax1.set_title(r'$S = A^T T_{trg}$')
    # ax2.boxplot(x2, notch=False, showmeans=False, showfliers=False)
    # ax2.set_xticks([])
    # ax2.set_title(r'$S_c = S - \overline{S}$')
    ax3.boxplot(x2, labels=texts[:t], notch=False, showmeans=False, showfliers=False)
    ax3.set_title(r'$\hat{S} = (I-P)S_c$')
    palette = ['#B9D2B1', '#33FFF1', '#F1D6B8', '#FBACBE', '#A8DADC', '#F9C74F','#FFF1E6', '#FFC6FF']
    xs1 = [np.random.normal(i + 1, 0.05, 6048) for i in range(t)]
    xs2 = xs1
    xs3 = xs1
    for i, (xA, xB, xC, c) in enumerate(zip(xs1, xs2, xs3, palette[:t])):
        valA, valB, valC = x1[:,i], x2[:,i], x3[:, i]
        ax1.scatter(xA, valA, alpha=0.4, color=c, s=2)
        # ax2.scatter(xB, valB, alpha=0.4, color=c, s=2)
        ax3.scatter(xC, valC, alpha=0.4, color=c, s=2)
    fig.show()
    plt.savefig('boxplot_s_dict.jpg', format='jpg')
create_df(x, dts, texts)