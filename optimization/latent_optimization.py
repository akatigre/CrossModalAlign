import os
import clip
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import math
from functools import partial

import argparse
from tqdm import tqdm

import torch
from torch import optim
import torch.nn.functional as F

from StyleCLIP.criteria.clip_loss import CLIPLoss
from StyleCLIP.criteria.id_loss import IDLoss

from StyleCLIP.models.stylegan2.model import Generator
from text_model import RandomInterpolation 
import wandb

l2norm = partial(F.normalize, p=2, dim=1)
STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]

STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in list(range(1, len(STYLESPACE_DIMENSIONS), 3))]


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def styleTransfer(args, device):
    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    mean_latent = g_ema.mean_latent(4096)

    if args.latent_path:
        idx = 5
        latent_code_init = torch.load(args.latent_path)[idx].to(device)
        
    elif args.mode == "edit":
        latent_code_init_not_trunc = torch.randn(1, 512).to(device)
        with torch.no_grad():
            _, latent_code_init, _ = g_ema([latent_code_init_not_trunc], return_latents=True,
                                        truncation=args.truncation, truncation_latent=mean_latent)
    else:
        latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)
        
        
    with torch.no_grad():
        img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)
    
    latent = latent_code_init.detach().clone()
    latent.requires_grad = True

    ################################################################################
    model, preprocess = clip.load('ViT-B/32', device)
    text_model = RandomInterpolation(512, model, preprocess, device, img_orig, args)
    text_model.to(device)

    clip_loss = CLIPLoss(args).to(device)
    idloss = IDLoss(args).to(device)

    trainable = [latent, text_model.edge_scaling, text_model.temperature] 
    optimizer = optim.Adam(trainable, lr=args.lr)
    pbar = tqdm(range(args.step))
    log_dict = {}
    text_embedding = text_model.text_feature
    
    if args.method != "baseline":
        image_manifold, gamma = text_model()
        text_star = l2norm(gamma * text_embedding + image_manifold)
    else:
        text_star = text_embedding

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
        new_image_feature = text_model.encode_image(img_gen)
        cs, us, ip = text_model.evaluation(new_image_feature)

        c_loss = clip_loss(img_gen, text_star)        
        l2_loss = ((latent_code_init - latent) ** 2).sum()
        ip_loss = (ip ** 2).sum()
        cs_loss = (cs ** 2).sum()
        log_dict["Core Semantic"] = cs_loss
        log_dict["Unwanted Semantic"] = us
        log_dict["Image Positive"] = ip_loss
        log_dict["clipLoss"] = c_loss
        log_dict["w+Loss"] = l2_loss
        loss = args.graph_lambda * c_loss + args.wplus_lambda * l2_loss + ip_loss
        log_dict["totalLoss"] = loss

        wandb.log(log_dict)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()


        pbar.set_description(
            (
                f"loss: {loss.item():.4f};"
            )
        )
    # Evaluation
    with torch.no_grad():
        final_result = torch.cat([img_orig, img_gen])
        id_loss = idloss(img_orig, img_gen)
    # file_name = f"/{args.latent_path.split('/')[-1][:-3]}_{args.description}.jpg"
    # path_name = os.path.join(args.results_dir, args.method)
    # gen_name = path_name+file_name
    # save_image(img_gen, gen_name, normalize=True, range=(-1, 1))# cosine similarity between generated image and text
    wandb.log({"Final Image": wandb.Image(final_result, caption=f"{args.method}"), "idloss": id_loss[0]})


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, default=None, help="the text that guides the editing/generation")
    parser.add_argument("--method", type=str)
    parser.add_argument("--ckpt", type=str, default="StyleCLIP/pretrained_models/stylegan2-ffhq-config-f.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--step", type=int, default=100, help="number of optimization steps")
    parser.add_argument("--batch_size", type=float, default=128)
    parser.add_argument("--mode", type=str, default="edit", choices=["edit", "free_generation"], help="choose between edit an image an generate a free one")
    parser.add_argument("--graph_lambda", type=float, default=1.0, help="weight of final embedding loss to minimize difference between text and reconstructed image")
    parser.add_argument("--wplus_lambda", type=float, default=0.008, help="weight of latent code in W+ space to prevent drastic change")
    parser.add_argument("--truncation", type=float, default=0.7, help="used only for the initial latent vector, and only when a latent code path is"
                                                                                                                    "not provided")
    parser.add_argument("--ir_se50_weights", type=str, default="StyleCLIP/pretrained_models/model_ir_se50.pth")
    parser.add_argument("--results_dir", type=str, default="results")
    

    args = parser.parse_args()

    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    import lpips
    lpips_alex = lpips.LPIPS(net='alex')
    args.lpips_alex = lpips_alex
    args.latent_path = "../mapper/test_faces.pt"
    if args.method!="baseline":
        args.wplus_lambda = 0.015
        args.method = "RandomInterpolation"
    else:
        args.wplus_lambda = 0.008
        args.method = "baseline"
    
    name = args.description
    wandb.init(project="Random Interpolation-Latent optimization", group=f"{args.description}", name = name, config={"method":args.method, "latent":args.latent_path, "w_lambda":args.wplus_lambda})
    styleTransfer(args, device)
    wandb.finish()
        
