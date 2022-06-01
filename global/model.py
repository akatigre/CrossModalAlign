import os
import sys
import numpy as np

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
)

import torch
from torch import arccos, nn
import torch.distributions as D
from numpy import linspace
from criteria.clip_loss import CLIPLoss
from criteria.id_loss import IDLoss
from utils.utils import l2norm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from scipy import stats

def kde_bivariate_plot(df):
    cl = ['b','y','r', 'g', 'm', 'k'] # Custom list of colours for each categories - increase as needed...

    headers = list(df.columns) # Extract list of column headers
    # Find min and max values for all x (= col [0]) and y (= col [1]) in dataframe:
    xmin, xmax = df.min(axis=0)[0], df.max(axis=0)[0]
    ymin, ymax = df.min(axis=0)[1], df.max(axis=0)[1]
    # Create a list of all unique categories which occur in the right hand column (ie index '2'):
    category_list = df.iloc[:,2].unique()

    # Set up 4 subplots and aspect ratios as axis objects using GridSpec:
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,3], height_ratios=[3,1])
    # Add space between scatter plot and KDE plots to accommodate axis labels:
    gs.update(hspace=0.3, wspace=0.3)

    fig = plt.figure() # Set background canvas colour to White instead of grey default
    fig.patch.set_facecolor('white')

    ax = plt.subplot(gs[0,1]) # Instantiate scatter plot area and axis range
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(headers[0], fontsize = 14)
    ax.set_ylabel(headers[1], fontsize = 14)
    ax.yaxis.labelpad = 5 # adjust space between x and y axes and their labels if needed
    ax.xaxis.labelpad = 0

    axl = plt.subplot(gs[0,0], sharey=ax) # Instantiate left KDE plot area
    axl.get_xaxis().set_visible(False) # Hide tick marks and spines
    axl.get_yaxis().set_visible(False)
    axl.spines["right"].set_visible(False)
    axl.spines["top"].set_visible(False)
    axl.spines["bottom"].set_visible(False)

    axb = plt.subplot(gs[1,1], sharex=ax) # Instantiate bottom KDE plot area
    
    axb.get_xaxis().set_visible(False) # Hide tick marks and spines
    axb.get_yaxis().set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.spines["top"].set_visible(False)
    axb.spines["left"].set_visible(False)

    axc = plt.subplot(gs[1,0]) # Instantiate legend plot area
    axc.axis('off') # Hide tick marks and spines

    # For each category in the list...
    for n in range(0, len(category_list)):
    # Create a sub-table containing only entries matching current category:
        st = df.loc[df[headers[2]] == category_list[n]]
        # Select first two columns of sub-table as x and y values to be plotted:
        x = st[headers[0]]
        y = st[headers[1]]

        # Plot data for each categorical variable as scatter and marginal KDE plots:    
        ax.scatter(x,y, color='none', s=100, edgecolor= cl[n], label = category_list[n])

        kde = stats.gaussian_kde(x, bw_method=0.1)
        xx = np.linspace(xmin, xmax, 1000)
        axb.plot(xx, kde(xx), color=cl[n])

        kde = stats.gaussian_kde(y, bw_method=0.1)
        yy = np.linspace(ymin, ymax, 1000)
        axl.plot(kde(yy), yy, color=cl[n])

    # Copy legend object from scatter plot to lower left subplot and display:
    # NB 'scatterpoints = 1' customises legend box to show only 1 handle (icon) per label 
    handles, labels = ax.get_legend_handles_labels()
    axc.legend(handles, labels, title = headers[2], scatterpoints = 1, loc = 'center', fontsize = 12)
    plt.savefig('plot.png', dpi=1800)
    plt.clf()

def bool2idx(b_list):
    tmp = [idx for idx, state in enumerate(b_list) if state]
    return tmp

def slerp(p0, p1, t):
    """
    p0, p1 : vector
    t : portion of the interpolation
    """
    omega = arccos(torch.dot((p0/torch.norm(p0)).squeeze(0), (p1/torch.norm(p1)).squeeze(0)))
    so = torch.sin(omega)
    return torch.sin((1.0-t)*omega)/so * p0 + torch.sin(t*omega)/so*p1

class CrossModalAlign(CLIPLoss):
    def __init__(self, args):
        super().__init__(opts=args)
        self.args = args
        # self.idloss = IDLoss(args).to(args.device)
        
    def cross_modal_surgery(self, fixed_weight=False):
        """
            self.text_feature and self.image_feature (in case of manipulation) should be assigned before call
        """
        # Target Text Dissection
        text_probs = (self.text_feature @ self.prototypes.T)
        df = self.break_down(text_probs)
        core_mask = np.array(df['categories']=='core')
        peri_mask = np.array(df['categories']=='peripheral')
        
        # core_mask, peri_mask = self.break_down(text_probs) # return in numpy arrays (6048, )

        # VERSION 1 : DIRECTLY MANIPULATE THE CHANNELS
        # Initialize the result array 
        m_idxs, m_weights = [], []

        # boolean array to index (which is True)
        core_mask, peri_mask = bool2idx(core_mask), bool2idx(peri_mask)

        core_semantics = self.prototypes[core_mask]
        weights =  self.text_feature @ core_semantics.T
        m_idxs.append(core_mask)
        if not fixed_weight:
            random_edges = D.relaxed_bernoulli.RelaxedBernoulli(probs=torch.abs(weights), temperature=torch.ones_like(weights))
            sampled_edges = random_edges.sample()
            weights = sampled_edges * torch.sign(weights)
        m_weights.extend(weights.detach().cpu().numpy())

        # PERIPHERAL
        peri_semantics = self.prototypes[peri_mask]
        weights = self.text_feature @ peri_semantics.T
        m_idxs.append(peri_mask)
        if not fixed_weight: 
            random_edges = D.bernoulli.Bernoulli(logits=torch.abs(weights))
            mask = random_edges.sample()
            weights = weights * mask
        m_weights.extend(weights.detach().cpu().numpy())
        
        # Image-related Units
        
        # image_probs = (self.image_feature @ self.prototypes.T)
        # c, p = self.break_down(image_probs) 
        # c, p = bool2idx(c), bool2idx(p)

        # img_mask = np.union1d(c, p)
        # txt_mask = np.union1d(core_mask, peri_mask)

        # overlap_mask = np.intersect1d(img_mask, txt_mask)
        # only_img_mask = np.setdiff1d(img_mask, overlap_mask)
        # filtered_mask = np.union1d(np.asarray([idx for idx in overlap_mask if image_probs.squeeze(0)[idx] * text_probs.squeeze(0)[idx]>=0]), only_img_mask)

        # img_weights = image_probs[:, filtered_mask]
        # m_idxs.append(filtered_mask)
        # m_weights.extend(img_weights.detach().cpu().numpy())

        return m_idxs, m_weights
     
    
    def projection(self, basis, target):
        B = basis.detach().cpu()
        X = target.detach().cpu()
        B = B.squeeze(0)
        X = X.squeeze(0)
        return l2norm((X.dot(B.T)/B.dot(B) * B).unsqueeze(0)).cuda()

    def break_down(self, probs, plot=False):
        clf = LocalOutlierFactor(algorithm='auto')
        probs = probs.T.cpu().detach().numpy()
        _ = clf.fit_predict(np.abs(probs))
        lof_score = -clf.negative_outlier_factor_
        kde = stats.gaussian_kde(np.abs(probs).flatten(), weights=lof_score)
        # kde = stats.gaussian_kde(np.vstack([lof_score.flatten(), probs.flatten()]),  bw_method='silverman')
        s1 = linspace(np.min(np.abs(probs)), np.max(np.abs(probs)), 100)
        # s2 = linspace(np.min(probs), np.max(probs), 100)
        kernel = kde(s1)
        # kde = KernelDensity(kernel='linear', bandwidth=0.01).fit(lof_score.reshape(-1, 1))
        
        # e = kde.score_samples(s.reshape(-1, 1)) # reshape a single feature
        mi = find_peaks(kernel)[0]
        print(mi)
        a, b = mi[-2], mi[-1]
        lof_score = lof_score.flatten()
        categories = ['unwanted' if c1 else 'peripheral' if c2 else 'core' for c1, c2 in zip(np.abs(probs) < s1[a],(np.abs(probs)>=s1[a])*(np.abs(probs)<s1[b]))]
        
        # core_mask, peri_mask = .squeeze(0), np.array().squeeze(0)
        df = pd.DataFrame(
            {
            'probs': np.abs(probs).flatten(), 
            'lof': np.log(lof_score).flatten(),
            'categories': categories,
            }
            )
        if plot:
            kde_bivariate_plot(df)
            exit()
        else:
            pass
        return df
        
        


    # def evaluation(self, img_orig, img_gen, target):
    #     """
    #     Evaluates manipulative quality in the generated image
    #     """
    #     # Identity Loss(ArcFace)
    #     if self.args.dataset != "AFHQ":
    #         identity = self.idloss(img_orig, img_gen)[0]
    #     else:
    #         identity = 0
            
    #     return identity

    def postprocess(self, random_text_feature):
        image_manifold = l2norm(self.image_semantics.sum(dim=0, keepdim=True))
        gamma = torch.abs(self.args.trg_lambda/(self.image_feature @ self.text_feature.T))
        text_star = gamma * random_text_feature + image_manifold
        img_prop = image_manifold.norm()/text_star.norm()
        return l2norm(text_star).detach().cpu().numpy(), img_prop
