import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
np.set_printoptions(suppress=True)
import wandb
import argparse

from model import CrossModalAlign

import torch
from torchvision.utils import save_image
from models.stylegan2.models import Generator

from utils import *
from utils.utils import *
from utils.stylegan_models import encoder, decoder
from utils.global_dir_utils import GetTemplate, GetBoundary, MSCode

from functools import partial

def generate_image(s_dict, t, args, style_space, style_names, alpha=5):
    boundary_tmp2, _, _, _ = GetBoundary(s_dict, t.squeeze(axis=0), args, style_space, style_names) # Move each channel by dStyle
    dlatents_loaded = [s.cpu().detach().numpy() for s in style_space]
    manip_codes= MSCode(dlatents_loaded, boundary_tmp2, [alpha], device)
    img_gen = decoder(generator, manip_codes, latent, noise_constants)
    return img_gen, manip_codes

def sequential_gen(s_dict, descriptions, args, initial_style_space, style_names, model):
    generated_images = []
    style_space = initial_style_space
    for target in descriptions:
        t = create_dt(target, model, "")
        t = t.detach().cpu().numpy()
        t = t/np.linalg.norm(t)
        img_gen, style_space = generate_image(s_dict, t, args, style_space, style_names)
        generated_images.append(img_gen)
    return generated_images

def create_dt(target, model, neutral=""):
    target_embedding = GetTemplate(target, model)- GetTemplate(neutral, model)
    t = target_embedding.unsqueeze(0).float()
    return t

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configuration for styleCLIP Global Direction with our method')
    parser.add_argument('--num_attempts', type=int, default=3, help="Number of iterations for diversity measurement")
    parser.add_argument('--topk', type=int, default=100, help="Number of channels to modify", choices=[25, 50, 100])
    parser.add_argument('--num_test', type=int, default=15)
    parser.add_argument('--trg_lambda', type=float, default=0.5, help="weight for preserving the information of target")
    parser.add_argument('--temperature', type=float, default=0.5, help="Used for bernoulli")
    parser.add_argument("--ir_se50_weights", type=str, default="../pretrained_models/model_ir_se50.pth")
    parser.add_argument("--stylegan_weights", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt")
    parser.add_argument("--segment_weights", type=str, default="./79999_iter.pth")
    parser.add_argument("--latents_path", type=str, default="../pretrained_models/test_faces.pt")
    parser.add_argument("--s_dict_path", type=str, default="./npy/ffhq/fs3.npy")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    args = parser.parse_args()
    args.ub = 0.3
    args.nsml = False
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    generator = Generator(
            size = 1024, # size of generated image
            style_dim = 512,
            n_mlp = 8,
            channel_multiplier = 2,
        )
        
    descriptions = ["asian", "big eyes"]


    generator.load_state_dict(torch.load(args.stylegan_weights, map_location='cpu')['g_ema'])
    generator.eval()
    generator.to(device)
 
    test_latents = torch.load(args.latents_path, map_location='cpu')
    subset_latents = torch.Tensor(test_latents[args.num_test:args.num_test+1, :, :]).cpu()

    
    s_dict = np.load(args.s_dict_path)
    align_model = CrossModalAlign(512, args)
    align_model.prototypes = torch.Tensor(s_dict).to(device)
    # args.s_dict = project_away_pc(s_dict, 5)
    args.s_dict = s_dict
    align_model.istr_prototypes = torch.Tensor(args.s_dict).to(device)
    
    align_model.to(device)
    
    latent = subset_latents[0].unsqueeze(0).to(device)
    imgs=[]
    with torch.no_grad():
        style_space, style_names, noise_constants = encoder(generator, latent)
        img_orig = decoder(generator, style_space, latent, noise_constants)
        align_model.image_feature = align_model.encode_image(img_orig)
        imgs.append(img_orig)
    seq_imgs = sequential_gen(s_dict, descriptions, args, style_space, style_names, align_model.model)
    imgs.extend(seq_imgs)
    align_model.text_feature = create_dt(' '.join(descriptions[-1]), align_model.model).to(device)
    align_model.cross_modal_surgery()
    random_text_feature = align_model.diverse_text()
    t, p = align_model.postprocess(random_text_feature)
    print(f"Image proportion {p}")
    am_img, _ = generate_image(args.s_dict, t, args, style_space, style_names, alpha=5)
    imgs.append(am_img)


    with torch.no_grad():
        img_name = f"Sequential-{args.num_test}-{' '.join(descriptions)}"
        img_dir = f"results/sequential/"
        os.makedirs(img_dir, exist_ok=True)
        imgs = torch.cat(imgs)
        save_image(imgs, f"{img_dir}/{img_name}.png", normalize=True, range=(-1, 1))
