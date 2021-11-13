######## Visualization of Cosine Similarities (Statistical) #########
#####################################################################
import matplotlib
from matplotlib import pyplot as plt
print(matplotlib.rcParams['text.usetex'])
from colour import Color
from scipy import stats
import statsmodels.api as sm
from utils.utils import project_away_pc
import os
import numpy as np
import torch
import clip
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils.global_dir_utils import create_dt
device =torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


texts = [
        "Grumpy",  "Asian",  "Female", "Smiling", "Lipstick", "Eyeglasses", \
        "friendly", "Bangs", "Black hair", "Blond hair", "Straight hair", "Bushy eyebrows", "Chubby", "Earrings", "Goatee", "Receding hairline", \
        "Grey hair", "Brown hair", "Wavy hair", "Wear suit", "Double chin", "Bags under eyes","Big nose", "Big lips",  "Old",\
        "Arched eyebrows","Muslim", "Tanned", "Pale", "Fearful", "He is feeling pressed","irresponsible", "inefficient", "intelligent", "Terrorist", "homocide", "handsome"
        ]


def plot_s_dict_box_plot(s_dict, texts):

    texts = [
        "Grumpy",  "Asian",  "Female", "Smiling", "Lipstick", "Eyeglasses", \
        "friendly", "Bangs", "Black hair", "Blond hair", "Straight hair", "Bushy eyebrows", "Chubby", "Earrings", "Goatee", "Receding hairline", \
        "Grey hair", "Brown hair", "Wavy hair", "Wear suit", "Double chin", "Bags under eyes","Big nose", "Big lips",  "Old",\
        "Arched eyebrows","Muslim", "Tanned", "Pale", "Fearful", "He is feeling pressed","irresponsible", "inefficient", "intelligent", "Terrorist", "homocide", "handsome"
        ]
    neutrals = [""]*len(texts)
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

def plot_top2PC(x, k=3, standard="ica"): # resolution or Independent Factor Analysis
    
    pca = PCA(n_components=k)
    mean, sd = x.mean(), x.std()
    print(mean, sd)
    x_tmp = (x - mean)/sd
    X_r = pca.fit(x_tmp).transform(x_tmp)
    fig = plt.figure(figsize=(5,5))
    if k==3:
        ax = fig.add_subplot(111, projection='3d')

    lw = 2
    red = Color("red")
    
    if standard=="resolution":
        target_names = [2**(1+((i+2)//2)) for i in range(18)][1:] # [4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]
        channels = [512, 512, 512, 512, 512, 256, 128, 64] * 2 + [32] # 총 17개
        y = []
        for idx, channel in enumerate(channels):
            tmp=[target_names[idx]]*channel
            y.extend(tmp) # [4] * 512 + [8] * 512 + [8] * 512 + ...
        y = np.array(y)
        t_n = [f"res{t}" for t in [4, 8, 16, 32, 64, 128, 256, 512, 1024]]
        labels = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        colors = list(red.range_to(Color("blue"), 9))
    else:
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x_tmp)
        y = kmeans.labels_
        labels = [i for i in range(n_clusters)]
        t_n=[f"cluster{i+1}" for i in range(n_clusters)]
        colors = list(red.range_to(Color("blue"), n_clusters))
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )
    
    mu = []
    for color, i, t in zip(colors, labels, t_n):
        if k==3:
            plt.scatter(
                X_r[y == i, 0], X_r[y == i, 1], X_r[y == i, 2], color=color.hex_l, alpha=0.8, lw=lw, label=t 
            )
        if k==2:
            plt.scatter(
                X_r[y == i, 0], X_r[y == i, 1], color=color.hex_l, alpha=0.8, lw=lw, label=t,s=1
            )
        mu.append(X_r[y == i,:].mean(axis=0))
    print(mu)
    if k==3:
        ax.set_xlabel("First Principal Component", fontsize=14)
        ax.set_ylabel("Second Principal Component", fontsize=14)
        ax.set_zlabel("Third Principal Component", fontsize=14)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of style space dictionary")
    plt.tight_layout()
    plt.savefig("PCA_s_dict.png")

s_dict = np.load(os.path.join("npy", "ffhq", "fs3.npy"))
x = project_away_pc(s_dict)
plot_top2PC(x,k=3)