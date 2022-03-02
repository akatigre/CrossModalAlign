import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import argparse
import numpy as np
np.set_printoptions(suppress=True)

from utils.utils import *
from utils.global_dir_utils import create_dt, manipulate_image, create_image_S
from model import CrossModalAlign
from models.stylegan2.model import Generator

from torchvision.utils import save_image

def prepare(args):
    # Load styleGAN generator
    generator = Generator(
        size = args.stylegan_size, # size of generated image  # ?
        style_dim = 512,
        n_mlp = 8,
        channel_multiplier = 2,
    )
    ckpt = torch.load(args.stylegan_weights, map_location='cpu')['g_ema']
    generator.load_state_dict(ckpt)
    generator.eval()
    generator.to(args.device)

    # Load anchors
    s_dict = np.load(args.s_dict_path)
    if args.method=="Random":
        args.s_dict = project_away_pc(s_dict, k=5)
    elif args.method=="Baseline":
        args.s_dict = s_dict

    align_model = CrossModalAlign(args)
    align_model.prototypes = torch.Tensor(args.s_dict).to(args.device)
    align_model.to(args.device)

    return generator, align_model, args


def run_global(generator, align_model, args, target, neutral):
    mean_latent = generator.mean_latent(4096)

    if args.random_latent:
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        # latent_code_init_not_trunc = torch.cat([torch.zeros(1, 511).cuda(),latent_code_init_not_trunc], dim=-1)
        with torch.no_grad():
            _, latent, _ = generator.forward([latent_code_init_not_trunc], return_latents=True,
                                        truncation=args.truncation, truncation_latent=mean_latent)
                                        # truncation=args.truncation, truncation_latent=latent_code_init_not_trunc)
        torch.save(latent, f'latent-{args.dataset}.pt')
    else:
        # latent = mean_latent.detach().clone().repeat(1, 18, 1)
        latent = torch.load(f"latent-{args.dataset}.pt")
    latent.to(args.device)
    
    target_embedding = create_dt(target, model=align_model.model, neutral=neutral)
    align_model.text_feature = target_embedding

    img_dir = f"{args.method}-{args.dataset}"
    os.makedirs(img_dir, exist_ok=True)
    
    generated_images = []
    # original Image from latent code (W+)
    img_orig, style_space, style_names, noise_constants = create_image_S(generator, latent)
    align_model.image_feature = align_model.encode_image(img_orig)
    generated_images.append(img_orig)

    for _ in range(args.num_attempts):
        # StyleCLIP GlobalDirection 
        if args.method=="Baseline":
            t = target_embedding.detach().cpu().numpy()
            t = t/np.linalg.norm(t)
        else:
            # Random Interpolation
            t = align_model.cross_modal_surgery()
        img_gen, _, _ = manipulate_image(style_space, style_names, noise_constants, generator, latent, args, alpha=5, t=t, s_dict=args.s_dict, device=args.device)
        generated_images.append(img_gen)

    img_name =  f"img-{args.method}-{target}"
    generated_images = torch.cat(generated_images) # [1+num_attempts, 3, 1024, 1024]
    save_image(generated_images, f"{img_dir}/{img_name}.png", normalize=True, range=(-1, 1))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configuration for styleCLIP Global Direction with our method')
    parser.add_argument('--method', type=str, default="Baseline", choices=["Baseline", "Random"], help='Use original styleCLIP global direction if Baseline')
    parser.add_argument('--num_attempts', type=int, default=3, help="Number of iterations for diversity measurement")
    parser.add_argument('--topk', type=int, default=50, help="Number of channels to modify", choices=[25, 50, 100])
    parser.add_argument('--trg_lambda', type=float, default=0.5, help="weight for preserving the information of target")
    parser.add_argument('--temperature', type=float, default=1.0, help="Used for bernoulli")
    parser.add_argument("--truncation", type=float, default=0.7, help="used only for the initial latent vector, and only when a latent code path is"                                                                "not provided")
    parser.add_argument("--stylegan_size", type=int, default=512, help="StyleGAN resolution for AFHQ")
    parser.add_argument("--nsml", action='store_true')
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--random_latent", action='store_true', help='activate if you want to use random latent')
    parser.add_argument("--dataset", type=str, default="afhqdog", choices=["afhqdog", "afhqcat", "afhqwild"])

    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    args.stylegan_weights = f"../pretrained_models/{args.dataset}.pt"
    args.s_dict_path = f"./npy/{args.dataset}/fs3.npy"
    
    generator, align_model, args = prepare(args)

    args.targets = ["kitten"]
    neutral = ["cat"]

    ###########################################################

    for idx, target in enumerate(args.targets):
        args.neutral = neutral[idx]
        run_global(generator, align_model, args, target, args.neutral)