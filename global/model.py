import os
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
from functools import partial
import scipy.stats as stats
from criteria.clip_loss import CLIPLoss
from criteria.id_loss import IDLoss
from utils.utils import projection, logitexp

l2norm = partial(F.normalize, p=2, dim=1)

class CrossModalAlign(CLIPLoss):
    def __init__(
        self,
        latent_size,
        args
    ):
        super().__init__(opts=args)
        self.edge_scaling = nn.Parameter(torch.tensor(1.0 * latent_size).sqrt().log())
        self.args = args
        self.idloss = IDLoss(args).cuda()
        
    def extract_image_positive(self):
        clip_sim = (self.image_feature @ self.prototypes.T).squeeze(0).detach().cpu()
        ip_mask = self.over_thres(clip_sim, alpha=1.0)
        self.image_cond = torch.stack([self.prototypes[idx] for idx in ip_mask if idx not in self.unwanted_mask])
        print(f"Source Positive {self.image_cond.shape[0]}")
        self.unwanted_mask = [i for i in self.unwanted_mask if i not in ip_mask]

    def disentangle_diverse_text(self, lb, ub):
        probs = (self.text_feature @ self.prototypes.T).squeeze(0).detach().cpu()
        tp_mask = self.over_thres(probs, alpha=lb)
        sc_mask = self.over_thres(probs, alpha=ub)
        self.unwanted_mask = [i for i in tp_mask if i not in sc_mask]
        
        if len(self.unwanted_mask)>0:
            print(f"Target core: {sc_mask} {len(sc_mask)} unwanted: {tp_mask} {len(tp_mask)}")
            self.core_cond = torch.stack([self.prototypes[idx] for idx in sc_mask])
            self.text_cond = torch.stack([self.prototypes[idx] for idx in self.unwanted_mask])
            random_core = self.diverse_text()
            # w = torch.stack([probs[i] for i in self.unwanted_mask]).unsqueeze(0).cuda()
            self.disentangled_text_feature = random_core - projection(basis=self.text_cond, target=random_core, multiple=True)
        else:
            lb += 0.05
            print("Nothing between the thresholds -> Decrease the lower bound")
            self.disentangle_diverse_text(lb=lb, ub=ub)

    def diverse_text(self):
        N = self.core_cond.shape[0]
        temp = torch.Tensor([self.args.temperature]*N).cuda()
        distances = (self.text_feature ** 2).sum(dim=1, keepdim=True) + (self.core_cond ** 2).sum(dim=1) - 2 * self.text_feature @ self.core_cond.T
        distances = - 0.5 * distances / self.edge_scaling.exp()
        edges = logitexp(distances.view(len(self.text_feature), len(self.core_cond)))
        random_edges = D.relaxed_bernoulli.LogitRelaxedBernoulli(logits=edges, temperature=temp)
        sampled_edges = random_edges.rsample()
        return torch.matmul(sampled_edges, self.core_cond)

    def over_thres(self, probs, alpha=1.0):
        sigma = probs.std()
        threshold = sorted(probs, reverse=True)[0] - alpha*sigma
        mask = (probs > threshold)
        indices = [i for i, b in enumerate(mask) if b]
        return indices
        
    def evaluation(self, img_orig, img_gen):
        """Evaluates manipulative quality & disentanglement in the generated image
        1. Core semantic: Increased (self.core_cond)
        2. Unwanted semantic: Do not increase (self.text_cond)
        3. Image positive: Do not decrease (self.image_cond)
        """
        # Identity Loss(ArcFace)
        identity = self.idloss(img_orig, img_gen)[0]
        new_image_feature = self.encode_image(img_gen)
        # Core semantic
        bf = self.image_feature @ self.core_cond.T
        af = new_image_feature @ self.core_cond.T
        cs = (af - bf).mean(dim=1)

        # Unwanted semantic (exclude anchors from image positive)
        conditions = torch.stack([self.prototypes[idx] for idx in self.unwanted_mask])
        bf = self.image_feature @ conditions.T
        af = new_image_feature @ conditions.T
        us = (af - bf).mean(dim=1)
        
        # Image Positive
        bf = self.image_feature @ self.image_cond.T
        af = new_image_feature @ self.image_cond.T
        ip = (af - bf).mean(dim=1)
        return identity, cs.detach().cpu().numpy(), us.detach().cpu().numpy(), ip.detach().cpu().numpy()

    def postprocess(self):
        weights = self.compute_edge_logits(self.image_feature.unsqueeze(0)[0], self.image_cond)
        image_manifold = l2norm(torch.mm(weights, self.image_cond))
        gamma = self.args.trg_lambda/(self.image_feature @ self.text_feature.T).mean()
        text_star = l2norm(gamma * self.disentangled_text_feature + image_manifold)
        return text_star

    def plot_hist(self, x, title: str):
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
