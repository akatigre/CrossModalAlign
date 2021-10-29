import os
import clip
from itertools import compress
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
from pathlib import Path
from functools import partial
import scipy.stats as stats


device = f"cuda:0" if torch.cuda.is_available() else "cpu"
latents_path = Path("latents")
proto_path = latents_path / "prototypes"

# with open(proto_path / "candidates.txt", 'r') as f:
#     candidates = []
#     for i in f.readlines():
#         candidates.append(i)
# PROTOTYPES = torch.load(proto_path / "attr2img_embeddings.pt").to(device).float()
# PROTOTYPES = torch.Tensor(np.load('./StyleCLIP/global/npy/ffhq/fs3.npy')).to(device) #6048, 512
PROTOTYPES = torch.Tensor(np.load('./StyleCLIP/global/npy/ffhq/W_manip_50comps_1000imgs.npy')).to(device) #900, 512
print(PROTOTYPES.shape)
l2norm = partial(F.normalize, p=2, dim=1)


class RandomInterpolation(nn.Module):
    def __init__(
        self,
        latent_size,
        model,
        preprocess,
        device,
        img_orig,
        args
    ):
        super().__init__()
        self.model, self.preprocess = model, preprocess
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size= 1024 // 32)
        self.edge_scaling = nn.Parameter(torch.tensor(1.0 * latent_size).sqrt().log())
        self.device = device
        self.args =args

        self.image_feature = self.encode_image(img_orig)
        if isinstance(args.description, str):
            self.text_feature = self.encode_text(args.description)
        else:
            self.text_feature = args.description
        self.extract_features()

    def extract_features(self):
        with torch.no_grad():
            self.extract_text_negative()
            self.extract_image_positive()
            n = len(self.image_cond)
            self.temperature = torch.Tensor([1]*n).to(self.device)

    def extract_image_positive(self):
        clip_sim = (self.image_feature @ PROTOTYPES.T).squeeze(0).detach().cpu()
        print(clip_sim.mean())
        print(clip_sim.std())
        # self.image_positive, ip_mask = self.over_quant(clip_sim, 5)
        ip_mask = self.over_quant(clip_sim, 5)
        self.image_cond = torch.stack([PROTOTYPES[idx] for idx in ip_mask])
        # print(f"Extract Image Positive: {self.image_positive}")

    def extract_text_negative(self):
        
        probs = (self.text_feature @ PROTOTYPES.T).squeeze(0).detach().cpu()
        print(probs.shape)
        mu, sigma = probs.mean(), probs.std()
        from matplotlib import pyplot as plt
        plt.hist(probs, density=True, alpha=0.2)
        plt.axvline(probs.mean(), label='mean')
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))   
        plt.title("Text Positives Probability")
        plt.savefig("TextPositive[fs3].png")
        plt.clf()
        # _, tp_mask = self.over_quant(probs, 5)
        tp_mask = self.over_quant(probs,5)
        # self.core_semantic, sc_mask = self.over_quant(probs, 1)
        sc_mask = self.over_quant(probs, 1)
        mask = [i for i in tp_mask if i not in sc_mask]
        self.core_cond = torch.stack([PROTOTYPES[idx] for idx in sc_mask])
        # self.unwanted = [candidates[idx] for idx in mask]
        self.text_cond = torch.stack([PROTOTYPES[idx] for idx in mask])
        # print(f"Extract Text Negative: {self.unwanted}")
        # print(f"Extract Core semantic: {self.core_semantic}")        
        # Prototype으로 제거
        
        w = torch.stack([probs[i] for i in mask]).unsqueeze(0).to(self.device)
        condition = l2norm(torch.mm(w, self.text_cond))
        self.text_feature -= self.projection(basis=self.text_feature, target=condition) # basis=condition, target=text feature do not work
       
        self.text_feature = l2norm(self.text_feature)

    def over_quant(self, probs, top):
        _, indices = torch.topk(probs, k=top)
        # top_list = list(map(candidates.__getitem__, indices))
        # return top_list, indices
        return indices
        
    def evaluation(self, new_image_feature):
        """Evaluates manipulative quality & disentanglement in the generated image
        1. Core semantic: Increased (self.core_cond)
        2. Unwanted semantic: Do not increase (self.text_cond)
        3. Image positive: Do not decrease (self.image_cond)

        """
        # Core semantic
        bf = self.image_feature @ self.core_cond.T
        af = new_image_feature @ self.core_cond.T
        cs = (af - bf).mean(dim=1)
        # Unwanted semantic
        bf = self.image_feature @ self.text_cond.T
        af = new_image_feature @ self.text_cond.T
        us = (af - bf).mean(dim=1)
        
        # Image Positive
        bf = self.image_feature @ self.image_cond.T
        af = new_image_feature @ self.image_cond.T
        ip = (af - bf).mean(dim=1)
        return cs, us, ip


    def forward(self):
        er = self.erdos_renyi(self.image_feature.unsqueeze(0), self.image_cond)
        weights = F.normalize(er, p=1, dim=1) # Interpolation
        # neighbors = l2norm(self.image_cond)
        image_manifold = torch.mm(weights, self.image_cond)
        gamma = 1/(self.image_feature @ self.text_feature.T)[0]
        return image_manifold, gamma
    

    def erdos_renyi(self, center, attrs):
        random_edges = self.compute_edge_logits(center[0], attrs)
        random_edges = D.relaxed_bernoulli.LogitRelaxedBernoulli(logits=random_edges, temperature=self.temperature)
        sampled_edges = random_edges.rsample()
        return sampled_edges

    def encode_text(self, text):
        tokenized = torch.cat([clip.tokenize(text)]).to(self.device)
        text_features = self.model.encode_text(tokenized.long())
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()

    def encode_image(self, img):
        img = self.avg_pool(self.upsample(img))
        image_features = self.model.encode_image(img)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        return image_features.float()

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

    def projection(self, basis, target, multiple=True):
        B = basis.detach().cpu()
        X = target.detach().cpu()
        
        if multiple:
            inv = torch.matmul(B, B.T)
            X = torch.matmul(B, X.T)
            P, _ = torch.solve(inv, X)
            proj = torch.matmul(P.T, B)
            return proj.to(self.device)
        else:
            X = X.squeeze(0)
            return (X.dot(B.T)/B.dot(B) * B).to(self.device)
        