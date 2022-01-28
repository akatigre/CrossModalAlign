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

    align_model = CrossModalAlign(512, args)
    align_model.prototypes = torch.Tensor(s_dict).to(args.device)
    align_model.istr_prototypes = torch.Tensor(args.s_dict).to(args.device) # Isotropic version of s_dict
    align_model.to(args.device)

    return generator, align_model, s_dict, args


def run_global(generator, align_model, args, target, neutral):
    mean_latent = generator.mean_latent(4096)

    path = False # if False, always use mean_latent
    if path:
        latent_code_init_not_trunc = torch.randn(1, 1).cuda()
        latent_code_init_not_trunc = torch.cat([torch.zeros(1, 511).cuda(),latent_code_init_not_trunc], dim=-1)
        with torch.no_grad():
            _, latent, _ = generator.forward([latent_code_init_not_trunc], return_latents=True,
                                        truncation=args.truncation, truncation_latent=mean_latent)
                                        # truncation=args.truncation, truncation_latent=latent_code_init_not_trunc)
        torch.save(latent, f'latent-{args.dataset}.pt')
    else:
        latent = mean_latent.detach().clone().repeat(1, 18, 1)
    latent.to(args.device)
    
    target_embedding = create_dt(target, model=align_model.model, neutral=neutral)
    align_model.text_feature = target_embedding
    
    if args.method == "Baseline":
        exp_name = f"{target}-chtopk{args.topk}"
    elif args.method == "Random":
        exp_name = f'{target}-chNum{args.topk}-Weight{args.trg_lambda}-Temp{args.temperature}'
    else:
        exp_name = None
        exit(-1)

    config = {
        "target": target,
        "method": args.method,
        "num channels": args.topk,
        "target weights": args.trg_lambda,
        "temperature": args.temperature,
    }
    group_name = args.method+"+LOF"
    if args.excludeImage:
        group_name += "-Image"
    if args.excludeRandom:
        group_name += "-Diverse"
    

    manip_channels = set()
    # latent = latent.unsqueeze(0).to(args.device)
    generated_images = []
    # original Image from latent code (W+)
    img_orig, style_space, style_names, noise_constants = create_image_S(generator, latent)
    align_model.image_feature = align_model.encode_image(img_orig)
    generated_images.append(img_orig)

    cs, ip, us, img_prop = [AverageMeter() for _ in range(4)]
    with torch.no_grad():
        # Prepare model for evaluation
        cores = align_model.cross_modal_surgery()

    for _ in range(args.num_attempts):
        # StyleCLIP GlobalDirection 
        if args.method=="Baseline":
            t = target_embedding.detach().cpu().numpy()
            t = t/np.linalg.norm(t)
        else:
        # Random Interpolation
            if not args.excludeRandom:
                random_cores = align_model.diverse_text(cores)
            else:
                random_cores = l2norm(align_model.core_semantics.sum(dim=0, keepdim=True))
            t, p = align_model.postprocess(random_cores)
            img_prop.update(p)
            
        img_gen, _, _ = manipulate_image(style_space, style_names, noise_constants, generator, latent, args, alpha=5, t=t, s_dict=args.s_dict, device=args.device)
        
        generated_images.append(img_gen)
        
        # Evaluation
        with torch.no_grad():
            _, _cs, _us, _ip, _ = align_model.evaluation(img_orig, img_gen, target)
            cs.update(_cs); us.update(_us); ip.update(_ip)

    img_name = f"{args.dataset}/{target}"
    generated_images = torch.cat(generated_images) # [1+num_attempts, 3, 1024, 1024]
    os.makedirs(group_name, exist_ok=True)
    save_image(generated_images, f"{group_name}/{img_name}.png", normalize=True, range=(-1, 1))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configuration for styleCLIP Global Direction with our method')
    parser.add_argument('--method', type=str, default="Baseline", choices=["Baseline", "Random"], help='Use original styleCLIP global direction if Baseline')
    parser.add_argument('--num_attempts', type=int, default=3, help="Number of iterations for diversity measurement")
    parser.add_argument('--topk', type=int, default=50, help="Number of channels to modify", choices=[25, 50, 100])
    parser.add_argument('--trg_lambda', type=float, default=0.5, help="weight for preserving the information of target")
    parser.add_argument('--temperature', type=float, default=1.0, help="Used for bernoulli")
    parser.add_argument("--truncation", type=float, default=0.7, help="used only for the initial latent vector, and only when a latent code path is"                                                                "not provided")
    parser.add_argument("--stylegan_size", type=int, default=512, help="StyleGAN resolution for AFHQ")
    parser.add_argument("--excludeImage", action='store_true', help="do not use image manifold information")
    parser.add_argument("--excludeRandom", action='store_true', help="do not use randomness of core semantics")
    parser.add_argument("--nsml", action='store_true')
    parser.add_argument("--gpu", type=int, default=0)
    
    parser.add_argument("--dataset", type=str, default="FFHQ", choices=["FFHQ", "afhqdog", "afhqcat", "afhqwild"])
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')

    if args.dataset != "FFHQ":
        args.s_dict_path = f"./npy/{args.dataset}/fs3.npy"
        args.stylegan_weights = f"../pretrained_models/{args.dataset}.pt"
    
    generator, align_model, s_dict, args = prepare(args)

    args.targets = ["cute"]
    neutral = [""]

    ###########################################################

    for idx, target in enumerate(args.targets):
        args.neutral = neutral[idx]
        run_global(generator, align_model, args, target, args.neutral)