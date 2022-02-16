import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import math
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial

import torch
from torch import optim
import torch.nn.functional as F
from torchvision.utils import save_image

from utils.utils import *
from model import CrossModalAlign
from models.stylegan2.models import Generator
import clip


l2norm = partial(F.normalize, p=2, dim=1)
STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]

STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in list(range(1, len(STYLESPACE_DIMENSIONS), 3))]



def prepare(args):
    
    # Load styleGAN generator
    generator = Generator(
        size = 1024, # size of generated image
        style_dim = 512,
        n_mlp = 8,
        channel_multiplier = 2,
    )
    generator.load_state_dict(torch.load(args.stylegan_weights, map_location='cpu')['g_ema'])
    generator.eval()
    generator.to(args.device)
  

    align_model = CrossModalAlign(512, args)
    a = torch.cuda.memory_allocated(0)
    align_model.to(args.device)
    s_dict = np.load("../global/npy/ffhq/fs3.npy")
    align_model.prototypes = torch.Tensor(s_dict).to(args.device) # Isotropic version of s_dict
    
    # text_inputs = torch.cat([clip.tokenize(args.target)]).to(args.device)
    text_feature = align_model.encode_text(args.target).to(args.device)
    align_model.text_feature = text_feature
    return generator, align_model, s_dict, args, text_feature

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def run_latent(args, idx, device):
    test_latents = torch.load(args.latent_path, map_location='cpu')[idx]
    
    # latent_code_init = test_latents[idx].unsqueeze(0).to(device)

    g_ema, align_model, _, args, text_feature = prepare(args)
    mean_latent = g_ema.mean_latent(4096)
    latent_code_init = torch.load('./boy.pt')
    # latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)
    
    if args.idx==1:
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
    with torch.no_grad():  
        if args.idx==1:
            _, latent_code_init = g_ema([latent_code_init_not_trunc], return_latents=True,
                                        truncation=0.7, truncation_latent=mean_latent)
        img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)
    
    diverse = []
    for attmpt in range(args.num_attempts):
        align_model.image_feature = align_model.encode_image(img_orig)
        
        latent = latent_code_init.detach().clone()
        latent.requires_grad = True

        trainable = [latent] 
        optimizer = optim.Adam(trainable, lr=args.lr)
        pbar = tqdm(range(args.step))
        
         
        with torch.no_grad():
            # Prepare model for evaluation
            cores = align_model.cross_modal_surgery()

        # StyleCLIP GlobalDirection 
        if args.method=="Baseline":
            t = text_feature.detach().cpu().numpy()
            t_star = t/np.linalg.norm(t)
        else:
        # Random Interpolation
            cores = align_model.diverse_text(cores)
            t_star, p = align_model.postprocess(cores)
        

        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False) 
            image_feature = align_model.encode_image(img_gen)
            c_loss = align_model(image_feature, torch.Tensor(t_star).to(device))
            l2_loss = ((latent_code_init - latent) ** 2).sum()
            
            loss = c_loss + args.wplus_lambda * l2_loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


            pbar.set_description(
                (
                    f"loss: {loss.item():.4f};"
                )
            )
        diverse.append(img_gen.detach().cpu())
    id_p = l2_loss.detach().cpu().item()
    print(f"{args.method}: {id_p}")
    img_name = f"{args.target}"
    return img_orig.detach().cpu(), diverse, img_name

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, help="the text that guides the editing/generation")
    parser.add_argument("--method", type=str, default="Baseline", choices=["Baseline", "Random"])
    parser.add_argument('--num_attempts', type=int, default=5, help="Number of iterations for diversity measurement")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--step", type=int, default=100, help="number of optimization steps")
    parser.add_argument("--batch_size", type=float, default=128)
    parser.add_argument("--wplus_lambda", type=float, default=0., help="weight of latent code in W+ space to prevent drastic change")
    parser.add_argument("--stylegan_weights", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--ir_se50_weights", type=str, default="../pretrained_models/model_ir_se50.pth")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--latent_path", type=str, default="../pretrained_models/test_faces.pt")
    parser.add_argument("--idx", type=int, default=0, help="Index of test latent face")
    parser.add_argument("--s_dict_path", type=str, default="../global/npy/ffhq/fs3.npy")
    parser.add_argument('--trg_lambda', type=float, default=0.5, help="weight for preserving the information of target")
    args = parser.parse_args()
    args.device=torch.device('cuda:0')
    original = args.wplus_lambda
    generated_images = []
    wplus = []
    for i, method in enumerate(["Baseline", "Random"]):
        args.method=method
        if args.method=="Baseline":
            args.wplus_lambda = 0.003
            wplus.append(args.wplus_lambda)
        else:
            args.wplus_lambda = 0.012
            wplus.append(args.wplus_lambda)
        img_orig, img_new, img_name = run_latent(args, args.idx, "cuda:0")
        if not i:
            generated_images.append(img_orig)
        generated_images.extend(img_new)   
    img_name += '-'.join([str(i) for i in wplus])
    img_dir = f"results/{args.target}/"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    save_image(torch.cat(generated_images), f"{img_dir}/{img_name}.png", normalize=True, range=(-1, 1))
        
