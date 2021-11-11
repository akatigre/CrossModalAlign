import os
import sys
import numpy as np

from CrossModalAlign.global.utils.eval_utils import Text2Prototype
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
)
import torch
from torch import nn
import torch.distributions as D
import scipy.stats as stats
from criteria.clip_loss import CLIPLoss
from criteria.id_loss import IDLoss
from utils.utils import projection, logitexp, l2norm
from utils.eval_utils import Text2Prototype

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
        
    def cross_modal_surgery(self):
        # Target Text Dissection
        text_probs = (self.text_feature @ self.prototypes.T)
        sc_mask = self.outlier_sigma(text_probs.squeeze(0), alpha=self.args.ub)
        print(f"Target core: {len(sc_mask)}-{sc_mask}")

        core_semantics = torch.stack([self.prototypes[idx] for idx in sc_mask])
        self.core_semantics = l2norm(torch.cat([self.text_feature, core_semantics]))
        # Source Image Dissection
        image_probs = (self.image_feature @ self.prototypes.T)
        ip_mask = self.outlier_sigma(image_probs.squeeze(0), alpha=1.0)
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

    def diverse_text(self):
        N = self.core_semantics.shape[0]
        temp = torch.Tensor([self.args.temperature]*N).cuda()
        cos_sim = self.text_feature @ self.core_semantics.T
        distances = (self.text_feature ** 2).sum(dim=1, keepdim=True) + (self.core_semantics ** 2).sum(dim=1) - 2 * self.text_feature @ self.core_semantics.T
        distances = - 0.5 * distances / self.edge_scaling.exp()
        edges = logitexp(distances.view(len(self.text_feature), len(self.core_semantics)))
        random_edges = D.relaxed_bernoulli.LogitRelaxedBernoulli(logits=edges, temperature=temp)
        sampled_edges = random_edges.rsample()
        weights = sampled_edges * torch.sign(cos_sim)
        diverse_core_manifold = torch.matmul(weights, self.core_semantics) # inner product
        return diverse_core_manifold # 1, 512

    def outlier_sigma(self, probs, alpha):
        sigma = probs.std()
        threshold_over = sorted(probs, reverse=True)[0] - alpha*sigma
        threshold_down = sorted(probs, reverse=False)[0] + alpha*sigma
        mask = (probs > threshold_over) | (probs < threshold_down)
        indices = [i for i, b in enumerate(mask) if b]
        return indices
        
    def evaluation(self, img_orig, img_gen, target):
        """Evaluates manipulative quality & disentanglement in the generated image
        1. Core semantic: Increased (self.core_semantics)
        2. Unwanted semantic: Do not increase (self.text_cond)
        3. Image positive: Do not decrease (self.image_semantics)
        """
        # Identity Loss(ArcFace)
        identity = self.idloss(img_orig, img_gen)[0]
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
            attr_prototype = torch.load(os.path.join('./prototypes-3', f"{attr}.pt")).cuda() # [512]
            # attr_neg = torch.load(os.path.join('./prototypes', f"{attr}_neg.pt")).cuda() # [512]
            # attr_prototype = projection(basis=self.text_feature, target=attr_prototype.float())
            # attr_orig_1 = self.image_feature @ attr_pos.float().T 
            # attr_orig_2 = self.image_feature @ attr_neg.float().T

            attr_orig = self.image_feature @ attr_prototype.T.float()
            # attr_orig = attr_orig_1 - attr_orig_2
            print(attr_orig)
            # print(attr_orig_2)
            attr_gen = new_image_feature @ attr_prototype.T.float()
            # attr_gen_1 = new_image_feature @ attr_pos.float().T# - new_image_feature @ attr_neg.float().T
            # attr_gen_2 = new_image_feature @ attr_neg.float().T
            print(attr_gen)
            # print(attr_gen_2)
            # attr_gen = attr_gen_1 - attr_gen_2

            # tmp = self.core_semantics @ attr_prototype.T.float()
            # print(tmp)
            attr = attr_gen - attr_orig
            print(attr.shape)
            attr = attr.detach().cpu().numpy()
        else: 
            attr = 0.0

        return identity, cs, abs(us), abs(ip), attr

    def postprocess(self, random_text_feature):
        image_manifold = self.image_semantics.sum(dim=0)
        gamma = torch.abs(self.args.trg_lambda/(self.image_feature @ self.text_feature.T))
        text_star = l2norm(gamma*random_text_feature + image_manifold)
        img_prop = image_manifold.norm()/(gamma * random_text_feature + image_manifold).norm()
        return text_star.detach().cpu().numpy(), img_prop

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
