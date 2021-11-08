import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import *
from utils.utils import *
import random
import argparse
import numpy as np
np.set_printoptions(suppress=True)

import wandb
import torch
from torchvision.utils import save_image
from model import CrossModalAlign
from models.stylegan2.models import Generator
from functools import partial

def gen_image_from_s(fs3, t, args, style_space, style_names, generator, latent, noise_constants):
    boundary_tmp2, _, _, _ = GetBoundary(fs3, t.squeeze(axis=0), args, style_space, style_names) # Move each channel by dStyle
    dlatents_loaded = [s.cpu().detach().numpy() for s in style_space]
    new_style_space = MSCode(dlatents_loaded, boundary_tmp2, [5.0], device="cuda:0")
    img_gen = decoder(generator, new_style_space, latent, noise_constants)
    return img_gen, new_style_space


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configuration for styleCLIP Global Direction with our method')
    parser.add_argument('--method', type=str, default="Baseline", choices=["Baseline", "Random"], help='Use original styleCLIP global direction if Baseline')
    parser.add_argument('--topk', type=int, default=25, help="Number of channels to modify")
    parser.add_argument('--lb', type=float, default=0.65, help="Threshold for unwanted semantics: elements with cosine similarity larger than lower bound")
    parser.add_argument('--ub', type=float, default=0.5, help="Threshold for unwanted semantics: elements with cosine similarity larger than lower bound")
    parser.add_argument('--trg_lambda', type=float, default=1.5, help="weight for preserving the information of target")
    parser.add_argument('--num_test', type=int, default=5, help="Number of test latents to use")
    parser.add_argument('--temperature', type=float, default=1.0, help="Used for bernoulli")
    parser.add_argument("--stylegan_weights", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt")
    parser.add_argument("--latents_path", type=str, default="../pretrained_models/test_faces.pt")
    parser.add_argument("--fs3_path", type=str, default="./npy/ffhq/fs3.npy")
    parser.add_argument("--wandb", action="store_true", help="Whether to activate the wandb")
    parser.add_argument("--textA", type=str, default="The man")
    parser.add_argument("--textB", type=str, default="Has no beard")
    parser.add_argument("--ir_se50_weights", type=str, default="../pretrained_models/model_ir_se50.pth")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")

    args = parser.parse_args()
    descriptions = [args.textA, args.textB]
    base_text = " ".join(descriptions)
    descriptions.append(base_text)
    if args.wandb:    
        exp_name = f'sequential-{args.textA}-{args.textB}'
        wandb.init(project="Global Direction Sequential Manipulation", name=exp_name)
    device = "cuda:0"
    
    generator = Generator(
            size = 1024, # size of generated image
            style_dim = 512,
            n_mlp = 8,
            channel_multiplier = 2
        )
    
    generator.load_state_dict(torch.load(args.stylegan_weights, map_location='cpu')['g_ema'])
    generator.eval()
    generator.to(device)

    fs3 = np.load(args.fs3_path)

    config = {
        "description_list": descriptions,
        "method": args.method,
        "num channels": args.topk,
        "target weights": args.trg_lambda,
    }

    
    align_model = CrossModalAlign(512, args)
    align_model.to(device)
    align_model.prototypes = torch.Tensor(fs3).to(device)
    subset_indices = random.sample([i for i in range(2824)], args.num_test)
    test_latents = torch.load(args.latents_path, map_location='cpu')[subset_indices, :, :]
    
    
    
    for latent_idx, latent in zip(subset_indices, test_latents):
        latent = latent.unsqueeze(0).to(device)
        imgs = []
        for idx, target in enumerate(descriptions):
            target_embedding = GetTemplate(target, align_model.model).unsqueeze(0).float()
            align_model.text_feature = target_embedding

            with torch.no_grad():
                if idx == 0:
                    style_space, style_names, noise_constants = encoder(generator, latent)
                    generate_img = partial(gen_image_from_s, fs3=fs3, args=args, style_names=style_names, generator=generator, latent=latent, noise_constants=noise_constants)
                    img_orig = decoder(generator, style_space, latent, noise_constants)
                    imgs.append(img_orig)

                align_model.image_feature = align_model.encode_image(img_orig)
                align_model.disentangle_diverse_text()
                align_model.extract_image_positive()

            # StyleCLIP GlobalDirection
            t = target_embedding.detach().cpu().numpy()
            t = t/np.linalg.norm(t)

            img_gen, new_style_space = generate_img(t=t, style_space=style_space)
            imgs.append(img_gen)

            if idx==1: # 1에서 2로 style space를 넘겨야함
                orig_style_space = style_space
                style_space = new_style_space # Update latent S
                orig_img = img_orig
                img_orig = img_gen

        # Use baseline target text to generate image with our method
        align_model.image_feature = align_model.encode_image(orig_img)
        align_model.disentangle_diverse_text()
        align_model.extract_image_positive()

        # OUR Method
        t = align_model.postprocess().detach().cpu().numpy()
        img_gen, new_style_space = generate_img(t=t, style_space=style_space)
        imgs.append(img_gen)


        with torch.no_grad():
            img_name = f"Original-{'-'.join(descriptions)}-OURS"
            img_dir = f"results/sequential/{args.method}/{exp_name}"

            os.makedirs(img_dir, exist_ok=True)
            imgs = torch.cat(imgs)
            save_image(imgs, f"{img_dir}/{img_name}.png", normalize=True, range=(-1, 1))
        if args.wandb:
            wandb.log({
                    "Generated image": wandb.Image(imgs, caption=img_name)})
    wandb.finish() 
