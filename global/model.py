import os
import sys
import numpy as np

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
)

import torch
from torch import nn
import torch.distributions as D
from numpy import linspace
from criteria.clip_loss import CLIPLoss
from criteria.id_loss import IDLoss
from utils.utils import l2norm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors.kde import KernelDensity


class CrossModalAlign(CLIPLoss):
    def __init__(self, args):
        super().__init__(opts=args)
        self.args = args
        self.idloss = IDLoss(args).to(args.device)
        
    def cross_modal_surgery(self):
        """
            self.text_feature and self.image_feature (in case of manipulation) should be assigned before call
        """
        # Target Text Dissection
        text_probs = (self.text_feature @ self.prototypes.T)
        core_mask, peri_mask = self.break_down(text_probs)

        # CORE
        core_semantics = self.prototypes[core_mask]
        weights =  self.text_feature @ core_semantics.T
        random_edges = D.relaxed_bernoulli.RelaxedBernoulli(probs=torch.abs(weights), temperature=torch.ones_like(weights))
        sampled_edges = random_edges.sample()
        weights = sampled_edges * torch.sign(weights)
        core_semantics = torch.matmul(weights, core_semantics)

        # PERIPHERAL
        peri_semantics = self.prototypes[peri_mask]
        weights = self.text_feature @ peri_semantics.T
        random_edges = D.bernoulli.Bernoulli(probs=torch.abs(weights))
        mask = random_edges.sample()
        peri_semantics = torch.matmul(weights*mask, peri_semantics)

        # Concatenate core + peripheral
        bases = l2norm(core_semantics+peri_semantics)

        # Source Image Dissection
        image_probs = (self.image_feature @ self.prototypes.T)
        c, p = self.break_down(image_probs)

        img_mask = np.union1d(c, p)
        txt_mask = np.union1d(core_mask, peri_mask)

        # Image positive filtering process
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
        core_mask, peri_mask = np.array([lof_score < s[a]]).squeeze(0), np.array([(lof_score>=s[a])*(lof_score<s[b])]).squeeze(0)
        return core_mask, peri_mask    


    def evaluation(self, img_orig, img_gen, target):
        """
        Evaluates manipulative quality in the generated image
        """
        # Identity Loss(ArcFace)
        if self.args.dataset != "AFHQ":
            identity = self.idloss(img_orig, img_gen)[0]
        else:
            identity = 0
            
        return identity

    def postprocess(self, random_text_feature):
        image_manifold = l2norm(self.image_semantics.sum(dim=0, keepdim=True))
        gamma = torch.abs(self.args.trg_lambda/(self.image_feature @ self.text_feature.T))
        text_star = gamma * random_text_feature + image_manifold
        img_prop = image_manifold.norm()/text_star.norm()
        return l2norm(text_star).detach().cpu().numpy(), img_prop
