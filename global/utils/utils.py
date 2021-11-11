from sklearn.decomposition import PCA
from torch.nn import functional as F
from functools import partial
import numpy as np
import torch

l2norm = partial(F.normalize, p=2, dim=-1)
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def projection(basis, target, multiple=False):
    B = basis.detach().cpu()
    X = target.detach().cpu()
    
    if multiple:
        inv_B = torch.solve(B, torch.matmul(B, B.T)).solution
        P = torch.matmul(B.T, inv_B)
        return l2norm(torch.matmul(X, P)).cuda()
    else:
        B = B.squeeze(0)
        X = X.squeeze(0)
        return l2norm((X.dot(B.T)/B.dot(B) * B).unsqueeze(0)).cuda()

def project_away_pc(x, k=5):
    pca = PCA(n_components=k)
    mean = x.mean()
    x_tmp = (x - mean)
    pca.fit(x_tmp)
    comp = np.matmul(np.matmul(x, pca.components_.T), pca.components_)
    return x_tmp - comp

def ffhq_style_semantic(channels):
    configs_ffhq = {
    'black hair' :      [(12, 479)],
    'blond hair':      [(12, 479), (12, 266)],
    'grey hair' :      [(11, 286)],
    'wavy hair'  :      [(6, 500), (8, 128), (5, 92), (6, 394), (6, 323)],
    'bangs'      :      [(3, 259), (6, 285), (5, 414), (6, 128), (9, 295), (6, 322), (6, 487), (6, 504)],
    'receding hairline':[(5, 414), (6, 322), (6, 497), (6, 504)],
    'smiling'    :      [(6, 501)],
    'lipstick'   :      [(15, 45)],
    'sideburns'  :      [(12, 237)],
    'goatee'     :      [(9, 421)],
    'earrings'   :      [(8, 81)],
    'glasses'    :      [(3, 288), (2, 175), (3, 120), (2, 97)],
    'wear suit'  :      [(9, 441), (8, 292), (11, 358), (6, 223)],
    'gender'     :      [(9, 6)]
    }
    style_channels = []
    for res, num_channels in channels.items():
        if res==4:
            style_channels.append(num_channels)
        else:
            style_channels.extend([num_channels]*2)
    
    mapped = {}
    for k, v in configs_ffhq.items():
        new_v = [sum(style_channels[:layer]) + ch for layer, ch in v]
        mapped[k] = new_v
    return mapped


def uniform_loss(x, t=2):
    x = torch.Tensor(x).cuda()
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def logitexp(logp):
    # Convert outputs of logsigmoid to logits (see https://github.com/pytorch/pytorch/issues/4007)
    pos = torch.clamp(logp, min=-0.69314718056)
    neg = torch.clamp(logp, max=-0.69314718056)
    neg_val = neg - torch.log(1 - torch.exp(neg))
    pos_val = -torch.log(torch.clamp(torch.expm1(-pos), min=1e-20))
    return pos_val + neg_val