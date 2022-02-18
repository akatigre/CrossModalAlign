import os
import sys
import numpy as np

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import torch.distributions as D
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from numpy import linspace
from sklearn.neighbors._kde import KernelDensity

from functools import partial
l2norm = partial(F.normalize, p=2, dim=-1)


class CrossModalAlign(object):
    def __init__(
        self,
        prototypes,
        args,
        text_feature,
        image_feature=None,
    ):
        super().__init__()
        self.text_feature = text_feature
        self.image_feature = image_feature
        self.prototypes = prototypes
        self.args = args

    def cross_modal_surgery(self):
        """[summary] Dissect the target text into text-relavant and image-relavant visual units

        Returns:
            [torch.tensor]: Set of text-relavant visual units filtered by source image
        """
        
        #! Text relavant visual units
        text_probs = (self.text_feature @ self.prototypes.T)
        core_mask, peri_mask = self.break_down(text_probs)

        #! Core
        core_semantics = self.prototypes[core_mask]
        weights = self.text_feature @ core_semantics.T
        random_edges = D.relaxed_bernoulli.RelaxedBernoulli(probs = torch.abs(weights), temperature=torch.ones_like(weights))
        sampled_edges = random_edges.sample()
        weights = sampled_edges * torch.sign(weights)
        core_semantics = torch.matmul(weights, core_semantics)

        #! Peripheral
        periph_semantics = self.prototypes[peri_mask]
        weights = self.text_feature @ periph_semantics.T
        random_edges = D.bernoulli.Bernoulli(probs = torch.abs(weights))
        mask = random_edges.sample()
        periph_semantics = torch.matmul(weights*mask, periph_semantics)
        bases = l2norm(core_semantics + periph_semantics)
        if self.image_feature is None:
            return bases
        else:
            #! Image relavant visual units
            image_probs = (self.image_feature @ self.prototypes.T)
            c, p = self.break_down(image_probs)
            img_mask = np.union1d(c, p)
            txt_mask = np.union1d(core_mask, peri_mask)

            #* Image positive filtering process
            only_img_mask = [i for i in img_mask if i not in txt_mask]
            overlap_mask = [i for i in img_mask if i in txt_mask]
            filtered_mask =[idx for idx in overlap_mask if image_probs.squeeze(0)[idx] * text_probs.squeeze(0)[idx]>=0] + only_img_mask
            img_proto = image_probs.T * self.prototypes
            image_manifold = l2norm(img_proto[filtered_mask].sum(dim=0, keepdim=True))
            gamma = torch.abs(1/(self.image_feature @ self.text_feature.T))
            return l2norm(gamma * bases + image_manifold)

    def projection(self, basis, target):
        B = basis.detach().cpu()
        X = target.detach().cpu()
        B = B.squeeze(0)
        X = X.squeeze(0)
        return l2norm((X.dot(B.T)/B.dot(B) * B).unsqueeze(0)).cuda()


    def break_down(self, probs):
        clf = LocalOutlierFactor(algorithm='auto', metric='cityblock')
        probs = probs.T.cpu().detach().numpy()
        _ = clf.fit_predict(probs)
        lof_score = clf.negative_outlier_factor_
        #! Kernel Density Estimation
        kde = KernelDensity(kernel='exponential', bandwidth=0.05).fit(lof_score.reshape(-1, 1))
        s = linspace(np.min(lof_score), np.max(lof_score))
        e = kde.score_samples(s.reshape(-1, 1)) # reshape a single feature
        from scipy.signal import find_peaks, peak_prominences
        mi = find_peaks(-e)[0]
        a, b = mi[0], mi[-1]
        plt.plot(s[:a+1], e[:a+1], 'r',
            s[a:b+1], e[a:b+1], 'b',
            s[b:], e[b:], 'm',
            s[[a,b]], e[[a,b]], 'y*')
        plt.scatter(lof_score, np.zeros_like(lof_score), c='c', alpha=0.8, marker='^')
        plt.xlabel('Local Outlier Factor')
        plt.ylabel('KDE score')
        plt.title("Binning Results")
        plt.savefig('kde.png')
        core_mask, peri_mask = np.array([lof_score < s[a]]).squeeze(0), np.array([(lof_score>=s[a])*(lof_score<s[b])]).squeeze(0)
        return core_mask, peri_mask
