import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import clip
import seaborn as sns
import torch
import math
from sklearn.decomposition import PCA
#* Boxplot

def project_away_pc(x, k=5):
    pca = PCA(n_components=k)
    mean = x.mean()
    x_tmp = (x - mean)
    pca.fit(x_tmp)
    comp = np.matmul(np.matmul(x, pca.components_.T), pca.components_)
    return x_tmp - comp
def boxplot():
    model, preprocess = clip.load("ViT-B/32", device="cuda")

    dataset = ['ffhq', 'car', 'church', 'afhqcat', 'afhqdog']
    texts = ['lipstick', 'Classic Car', 'Gothic Church', 'Cute Cat', 'Bulldog']
    df = []
    for t, d in zip(texts, dataset):
        p = np.load(f'./stylespace/{d}.npy') # 6048, 512
        p = torch.Tensor(project_away_pc(p)).cuda()
        text = clip.tokenize(t).cuda() #tokenize
        text_embeddings = model.encode_text(text).float() #embed with text encoder
        c = (p @ text_embeddings.T).detach().cpu().numpy()
        df.append(c)
    vals, names, xs = [], [], []
    for i, sim in enumerate(df):
        vals.append((sim-sim.mean())/sim.std())
        xs.append(np.random.normal(i+1, 0.04, sim.shape[0]))
    cossim = np.asarray(vals)
    plt.boxplot(cossim, labels=texts, notch=True, boxprops=dict(linestyle='-', linewidth=1.5, color='#00145A'))
    cmap = matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap("Pastel1").colors[:10])
    mp = dict(zip(list(set(dataset)), list(range(5))))
    for i, (x, val) in enumerate(zip(xs, vals)):
        c = cmap.colors[mp[dataset[i]]]
        plt.scatter(x, val, alpha=0.3, s=0.5, color=c)
    for i, sim in enumerate(vals):
        c = cmap.colors[mp[dataset[i]]]
        hist = np.histogram(sim, bins=50)
        h = math.ceil(max(hist[1])) - math.floor(min(hist[1]))
        h /= len(hist[0])
        plt.barh(hist[1][:-1], hist[0]/500, height=h, left=1+i, alpha=0.4, color = c)
    plt.savefig('boxplot.png', dpi=1200)

if __name__ == '__main__':
    boxplot()

