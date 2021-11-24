import os
import sys
import numpy as np

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
)
from itertools import compress
import torch
from torch import nn
import torch.distributions as D
import scipy.stats as stats
from criteria.clip_loss import CLIPLoss
from criteria.id_loss import IDLoss
from utils.utils import l2norm
from utils.eval_utils import Text2Prototype
from sklearn.neighbors import LocalOutlierFactor
import ckwrap


class CrossModalAlign(CLIPLoss):
    def __init__(
        self,
        latent_size,
        args
    ):
        super().__init__(opts=args)
        self.edge_scaling = nn.Parameter(torch.tensor(1.0 * latent_size).sqrt().log())
        self.args = args
        self.idloss = IDLoss(args).to(args.device)
        
    def cross_modal_surgery(self):
        # Target Text Dissection
        text_probs = (self.text_feature @ self.prototypes.T)
        sc_mask, (core_mask, perip_mask) = self.outlier_sigma(text_probs.squeeze(0), cnt=0.5)
        # self.core_probs = torch.stack([text_probs.squeeze(0)[idx] for idx in sc_mask])
        print(f"Target core: {len(sc_mask)}-{sc_mask} Core {core_mask} Peripheral {perip_mask}")
        self.core_mask = np.where(sc_mask == core_mask)
        self.core_semantics = l2norm(torch.stack([text_probs.squeeze(0)[idx] * self.prototypes[idx] for idx in sc_mask]))
        # Source Image Dissection
        image_probs = (self.image_feature @ self.prototypes.T)
        ip_mask, _ = self.outlier_sigma(image_probs.squeeze(0),cnt=0.1)

        #* Image positive filtering process
        only_img_mask = [i for i in ip_mask if i not in sc_mask]
        overlap_mask = [i for i in ip_mask if i in sc_mask] # 겹치는 것 중에서 방향이 일치하는 경우만 포함  
        ip_mask =[idx for idx in overlap_mask if image_probs.squeeze(0)[idx] * text_probs.squeeze(0)[idx]>=0] + only_img_mask
        img_proto = image_probs.T * self.prototypes
        self.image_semantics = l2norm(img_proto[ip_mask])
        print(f"Source Positive {self.image_semantics.shape[0]}")

        # Unwanted semantics from text should exclude core semantics and image positives
        unwanted_mask = [i for i in range(image_probs.shape[0]) if i not in ip_mask+sc_mask]
        txt_proto = text_probs.T * self.prototypes
        self.unwanted_semantics = l2norm(txt_proto[unwanted_mask])
        return l2norm(torch.stack([self.prototypes[idx] for idx in sc_mask])) 

    def diverse_text(self, cores):
        N = cores.shape[0]
        cos_sim = self.text_feature @ cores.T
        random_edges = D.bernoulli.Bernoulli(probs = torch.abs(cos_sim))
        sampled_edges = random_edges.sample()
        print(f"sampled :{torch.count_nonzero(sampled_edges)}")
        sampled_edges[self.core_mask] = 1
        weights = (cos_sim * sampled_edges) * torch.sign(cos_sim)
        diverse_core_manifold = l2norm(torch.matmul(weights, cores)) # inner product
        return diverse_core_manifold
        

    def outlier_sigma(self, probs, alpha=1.96, cnt=0.1):
        mu, sigma = probs.mean(), probs.std()
        threshold_over = mu + alpha*sigma
        threshold_down = mu - alpha*sigma
        m1 = torch.ge(probs, threshold_over)
        m2 = torch.le(probs, threshold_down)
        mask = torch.bitwise_or(m1, m2)
        candidate_indices = [i for i, b in enumerate(mask) if b]
        candidate_probs = probs[mask].detach().cpu().numpy().reshape(-1, 1) # (N, 1)
        
        clf = LocalOutlierFactor(n_neighbors=5, contamination=cnt)
        y_pred = clf.fit_predict(candidate_probs)
        outlier_mask = (y_pred==-1)
        indices = list(compress(candidate_indices, outlier_mask.tolist()))
        lof_score = clf.negative_outlier_factor_[outlier_mask]
        km = ckwrap.ckmeans(lof_score, 2)
        buckets = [[],[]]
        for i in range(len(lof_score)):
            buckets[km.labels[i]].append(indices[i])
        return indices, buckets

        
    def evaluation(self, img_orig, img_gen, target):
        """Evaluates manipulative quality & disentanglement in the generated image
        1. Core semantic: Increased (self.core_semantics)
        2. Unwanted semantic: Do not increase (self.text_cond)
        3. Image positive: Do not decrease (self.image_semantics)
        """
        # Identity Loss(ArcFace)
        if self.args.dataset != "AFHQ":
            identity = self.idloss(img_orig, img_gen)[0]
        else:
            identity = 0
            
        new_image_feature = self.encode_image(img_gen)

        # Core semantic
        bf = self.image_feature @ self.core_semantics.T
        af = new_image_feature @ self.core_semantics.T
        cs = (af - bf).mean(dim=1)
        cs = cs.detach().cpu().numpy()

        # Unwanted semantic (exclude anchors from image positive)
        
        bf = self.image_feature @ self.unwanted_semantics.T
        af = new_image_feature @ self.unwanted_semantics.T
        us = (af - bf).mean(dim=1)
        us = us.detach().cpu().numpy()
    
        if self.image_semantics.shape[0] == 0: 
            ip = 0.0
        else: 
            # Image Positive
            bf = self.image_feature @ self.image_semantics.T
            af = new_image_feature @ self.image_semantics.T
            ip = (af - bf).mean(dim=1)
            ip = ip.detach().cpu().numpy()

        attr = Text2Prototype(target)
        if attr != None: 
            attr_prototype = torch.load(os.path.join('./prototypes-3', f"{attr}.pt")).to(self.args.device)
            attr_orig = self.image_feature @ attr_prototype.T.float()
            attr_gen = new_image_feature @ attr_prototype.T.float()
            attr = attr_gen - attr_orig
            attr = attr.detach().cpu().numpy()
        else: 
            attr = 0.0

        return identity, cs, abs(us), abs(ip), attr

    def postprocess(self, random_text_feature):
        image_manifold = l2norm(self.image_semantics.sum(dim=0, keepdim=True))
        gamma = torch.abs(self.args.trg_lambda/(self.image_feature @ self.text_feature.T))
        text_star = gamma * random_text_feature + image_manifold
        img_prop = image_manifold.norm()/text_star.norm()
        return l2norm(text_star).detach().cpu().numpy(), img_prop

    def check_normal(self, x, title: str):
        from matplotlib import pyplot as plt
        plt.hist(x, bins=100, density=True, stacked=True, alpha=0.2)
        mu, sigma = x.mean(), x.std()
        plt.axvline(x.mean(), label='mean')
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.text(-0.2, 5, f"mean: {mu} std: {sigma}")
        plt.plot(x, stats.norm.pdf(x, mu, sigma))
        plt.grid(True)
        plt.title(f"{title}")
        if not os.path.exists('results/'):
            os.mkdir('results/')
        plt.savefig(f"results/{title}.png")
        plt.clf()
