import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import wandb
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

    # Load anchors
    s_dict = np.load("../global/npy/ffhq/fs3.npy")

    align_model = CrossModalAlign(512, args)
    align_model.prototypes = torch.Tensor(s_dict).to(args.device)
    align_model.istr_prototypes = torch.Tensor(s_dict).to(args.device) # Isotropic version of s_dict
    align_model.to(args.device)

    return generator, align_model, s_dict, args

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def run_latent(args, idx, device):
    test_latents = torch.load(args.latent_path, map_location='cpu')
    latent_code_init = test_latents[idx].to(device)
    g_ema, align_model, _, args = prepare(args)

    generated_images = []
    with torch.no_grad():
        img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)
        align_model.image_feature = align_model.encode_image(img_orig)
    generated_images.append(img_orig.detach().cpu())
    target_embedding = align_model.encode_text(args.target)

    for attmpt in args.num_attempts:
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
            t = target_embedding.detach().cpu().numpy()
            t_star = t/np.linalg.norm(t)
        else:
        # Random Interpolation
            if not args.excludeRandom:
                cores = align_model.diverse_text(cores)
            else:
                cores = l2norm(align_model.core_semantics.sum(dim=0, keepdim=True))
            t_star, p = align_model.postprocess(cores)

        

        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
            
            with torch.no_grad():
                identity, cs, us, ip = align_model.evaluation(img_orig, img_gen)

            c_loss = align_model(img_gen, t_star)
            l2_loss = ((latent_code_init - latent) ** 2).sum()
            
            
            loss = args.graph_lambda * c_loss + args.wplus_lambda * l2_loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


            pbar.set_description(
                (
                    f"loss: {loss.item():.4f};"
                )
            )
        generated_images.append(img_gen.detach().cpu())
        # Evaluation
        
        # wandb.log({
        #         "Generated image": wandb.Image(imgs, caption=args.method),
        #         "core semantic": np.round(cs, 3), 
        #         "unwanted semantics": np.round(us, 3), 
        #         "source positive": np.round(ip, 3),
        #         "identity loss": identity,
        #         })
        img_name =  f"img{args.idx}-{args.method}-{args.target}-{attmpt}"


        save_image(torch.cat(generated_images), f"results/{img_name}.png", normalize=True, range=(-1, 1))

    wandb.finish()
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, help="the text that guides the editing/generation")
    parser.add_argument("--method", type=str, choices=["Baseline", "Random"])
    parser.add_argument('--num_attempts', type=int, default=1, help="Number of iterations for diversity measurement")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--step", type=int, default=100, help="number of optimization steps")
    parser.add_argument("--batch_size", type=float, default=128)
    parser.add_argument("--wplus_lambda", type=float, default=0.008, help="weight of latent code in W+ space to prevent drastic change")
    parser.add_argument("--stylegan_weights", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--ir_se50_weights", type=str, default="../pretrained_models/model_ir_se50.pth")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--latent_path", type=str, default="../pretrained_models/test_faces.pt")
    parser.add_argument("--idx", type=int, default=0, help="Index of test latent face")
    parser.add_argument("--args.s_dict_path", type=str, default="../global/npy/ffhq/fs3.npy")
    args = parser.parse_args()
    args.device=torch.device('cuda:0')
    if args.method=="Random":
        args.wplus_lambda = 0.015
        args.method = "RandomInterpolation"
    elif args.method=="Baseline":
        args.wplus_lambda = 0.008
        args.method = "baseline"
        exp_name = f"method{args.method}-chNum{args.topk}"
    else:
        NotImplementedError()
    
    config = {
        "method": args.method,
        "target": args.target
    }
 
    run_latent(args, args.idx, "cuda:0")
    
        
