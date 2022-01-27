import seaborn as sns
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from colour import Color


if __name__ == "__main__2":
    parser = argparse.ArgumentParser(description='Configuration for styleCLIP Global Direction with our method')
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    args = parser.parse_args()
    args.device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')

    s_dict = np.load("./npy/ffhq/fs3.npy")

    original = torch.load('orig.pt').cpu()
    original = torch.mean(original, dim=0).unsqueeze(0)

    text_list = [
   "Grumpy",  "Asian",  "Female", "Smiling", "Lipstick", "Eyeglasses", \
    "friendly", "Bangs", "Black hair", "Blond hair", "Straight hair", "Bushy eyebrows", "Chubby", "Earrings", "Goatee", "Receding hairline", \
    "Grey hair", "Brown hair", "Wavy hair", "Wear suit", "Double chin", "Bags under eyes","Big nose", "Big lips",  "Old",\
    "Arched eyebrows","Muslim", "Tanned", "Pale", "Fearful", "He is feeling pressed","irresponsible", "inefficient", "intelligent", "Terrorist", "homocide", "handsome", \
    "African American", "Albino", "Attractive", "Black", "blue-eyed", "brown-eyed", "brunette", "feminine", "frizzy", "frumpy", "Indian", "Italian", "Lanky", "large", "masculine",\
    "muscular", "painted", "petite", "pierced", "polished", "rosy"]

    clip_loss = CLIPLoss(args)
    text_grey = clip_loss.encode_text("Grey").cpu().detach()
    # probs = np.dot(s_dict, text_grey)
    # _, idxs = torch.topk(torch.Tensor(np.abs(probs)), 1)
    # text_grey = torch.Tensor(s_dict[idxs]).unsqueeze(0)

    text_old = clip_loss.encode_text("Old").cpu().detach()
    text_stack = []
    for text in text_list: 
        tmp = clip_loss.encode_text(text).cpu().detach()
        text_stack.append(tmp)
    text_stack = torch.cat(text_stack, dim=0)
    
    gen1 = torch.load('grey-1.pt').cpu()
    gen1 = torch.mean(gen1, dim=0).unsqueeze(0)
    gen2 = torch.load('grey-2.pt').cpu()
    gen2 = torch.mean(gen2, dim=0).unsqueeze(0)

   
    pca = PCA(n_components=2)
    x = torch.cat([gen1, gen2, original, text_old, text_grey, text_stack], dim=0)
    mean, sd = x.mean(), x.std()
    x_tmp = (x - mean)/sd
    X_r = pca.fit(x_tmp).transform(x_tmp)
    fig = plt.figure(figsize=(5,5))

    lw = 2
    red = Color("red")
    t = ['CLIP(Grey)', 'CLIP(Grey)-CLIP(OLD)', 'Original', 'Text_Old', 'Text_Grey']
    labels = [0, 1, 2, 3, 4]
    colors = list(red.range_to(Color("blue"), 5))
    # y = [0]* 50 + [1] * 50 + [2]* 50 +[3, 4]
    y = [0,1,2,3,4] + [-1]*58
    y = np.array(y)

    mu = []
    for color, i, t in zip(colors, labels, t):
        plt.scatter(
            X_r[y == i, 0], X_r[y == i, 1], color=color.hex_l, alpha=0.8, lw=lw, label=t, s=1
        )
        mu.append(X_r[y == i,:].mean(axis=0))
    
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.tight_layout()
    plt.savefig("plot2.png")


if __name__ == "__main__":

    #61 제거 78 제거

    df = pd.read_csv('plot_histogram.csv')
    sns.set_theme(style='whitegrid')
    # palette = sns.color_palette('Set2')
    # sns.set_palette(palette)

    # create plot
    ax = sns.barplot(data=df, x="Text", y="average_cnt", hue="Method")

    # for p in ax.patches: 
    #     ax.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width()/2., p.get_height() - 30), 
    #     ha='center', va='center', fontsize=10, color='black', xytext=(0, 10), 
    #     textcoords='offset points') 
    my_ticks = ["Blond\n hair","Bangs","Eye\nglasses","Smiling","Young", "Wavy\nHair","Receding\nHairline","Big\nLips"]
    ax.set_xticklabels(my_ticks)
    ax.set(xlabel=None)
    plt.ylabel('Manipulation Accuracy')
    plt.title("Target-wise Manipulation Performance")
    
    plt.savefig('barplot.png', dpi =300)
    