import os
import sys
import torch
import argparse
import numpy as np
np.set_printoptions(suppress=True)

import wandb


from utils.utils import *
from utils.global_dir_utils import create_dt, manipulate_image, create_image_S
from criteria.clip_loss import CLIPLoss
import sys
sys.path.append('../../cosmos')
from cosmos.model import CrossModalAlign
from models.stylegan2.models import Generator

from tqdm import tqdm
from torchvision.utils import save_image




def run_global(generator, align_model, args, target, neutral):
    generator = Generator(
        size = 1024,
        style_dim = 512,
        n_mlp = 8,
        channel_multiplier = 2,
    )
    generator.load_state_dict(torch.load(args.stylegan_weights, map_location='cpu')['g_ema'])
    generator.eval()
    generator.to(args.device)

    # Load anchors
    s_dict = np.load(args.s_dict_path)
    if args.method=="Random":
        s_dict = project_away_pc(s_dict, k=5)

    test_latents = torch.load(args.latents_path, map_location='cpu')
    text_feature = create_dt(target, model=align_model.model, neutral=neutral)

    img_orig, style_space, style_names, noise_constants = create_image_S(generator, latent, args)
    image_feature = CLIPLoss.encode_image(img_orig)
    
    align_model = CrossModalAlign(prototypes=s_dict, args=args, text_feature=text_feature, image_feature=image_feature)
    align_model.prototypes = torch.Tensor(args.s_dict).to(args.device)
    align_model.to(args.device)
    generated_images = []
    for _ in tqdm(range(args.num_attempts)):
        with torch.no_grad():
            if args.method=="Baseline":
                t = text_feature.detach().cpu().numpy()
                t = t/np.linalg.norm(t)
            else:
                random_cores = align_model.diverse_text()
                t = align_model.cross_modal_surgery()
        img_gen, _, _ = manipulate_image(style_space, style_names, noise_constants, generator, latent, args, alpha=10, t=t, s_dict=s_dict, device=args.device)
        generated_images.append(img_gen.detach().cpu())
        
    generated_images = torch.cat(generated_images) # [1+num_attempts, 3, 1024, 1024]
    save_image(generated_images, f"results/{args.dataset}/{target}-{args.method}-diverse.png", normalize=True, range=(-1, 1))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configuration for styleCLIP Global Direction with our method')
    parser.add_argument('--method', type=str, default="Baseline", choices=["Baseline", "Random"], help='Use original styleCLIP global direction if Baseline')
    parser.add_argument('--num_attempts', type=int, default=10, help="Number of iterations for diversity measurement")
    parser.add_argument('--topk', type=int, default=50, help="Number of channels to modify")
    parser.add_argument("--ir_se50_weights", type=str, default="../pretrained_models/model_ir_se50.pth")
    parser.add_argument("--stylegan_weights", type=str, default="../pretrained_models/FFHQ.pt")
    parser.add_argument("--segment_weights", type=str, default="../pretrained_models/79999_iter.pth")
    parser.add_argument("--latents_path", type=str, default="test_latent.pt")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    args.s_dict_path = f"./npy/ffhq/fs3.npy"
    args.stylegan_weights = f"../pretrained_models/ffhq.pt"
    
    args.targets = ['big nose', 'small nose']

    neutral = [""]*len(args.targets)

    for idx, target in enumerate(args.targets):
        args.neutral = neutral[idx]
        run_global(generator, align_model, args, target, args.neutral)