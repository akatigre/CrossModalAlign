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
from utils.utils import projection

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
        return ip_mask

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
            gamma = torch.abs(1/(self.image_feature @ self.text_feature.T).mean())
            print(gamma)
            self.disentangled_text_feature = gamma * random_core - projection(basis=self.text_cond, target=random_core, multiple=True)
            return self.unwanted_mask, sc_mask
        else:
            lb += 0.05
            print("Nothing between the thresholds -> Decrease the lower bound")
            return self.disentangle_diverse_text(lb=lb, ub=ub)
            

    def diverse_text(self):
        N = self.core_cond.shape[0]
        temp = torch.Tensor([self.args.temperature]*N).cuda()
        weight = self.erdos_renyi(self.text_feature.unsqueeze(0), self.core_cond, temp)
        return torch.matmul(weight, self.core_cond)

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
        # gamma = 1/(self.image_feature @ self.text_feature.T)[0]
        text_star = l2norm(self.args.trg_lambda * self.disentangled_text_feature + image_manifold)
        return text_star

    def erdos_renyi(self, center, attrs, temp):
        random_edges = self.compute_edge_logits(center[0], attrs)
        random_edges = D.relaxed_bernoulli.LogitRelaxedBernoulli(logits=random_edges, temperature=temp)
        sampled_edges = random_edges.rsample()
        return sampled_edges

    
    def compute_edge_logits(self, center, attrs):
        def logitexp(logp):
            # Convert outputs of logsigmoid to logits (see https://github.com/pytorch/pytorch/issues/4007)
            pos = torch.clamp(logp, min=-0.69314718056)
            neg = torch.clamp(logp, max=-0.69314718056)
            neg_val = neg - torch.log(1 - torch.exp(neg))
            pos_val = -torch.log(torch.clamp(torch.expm1(-pos), min=1e-20))
            return pos_val + neg_val
        distances = (center ** 2).sum(dim=1, keepdim=True) + (attrs ** 2).sum(dim=1) - 2 * center @ attrs.T
        distances = - 0.5 * distances / self.edge_scaling.exp()
        logits = logitexp(distances.view(len(center), len(attrs)))
        return logits

    
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
