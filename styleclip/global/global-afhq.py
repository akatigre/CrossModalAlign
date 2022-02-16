import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import argparse
import numpy as np
np.set_printoptions(suppress=True)

from utils.utils import *
from utils.global_dir_utils import create_dt, manipulate_image, create_image_S
from cosmos.model import CrossModalAlign
from models.stylegan2.model import Generator_
from models.stylegan2.models import Generator
from torchvision.utils import save_image

def prepare(args):
    # Load styleGAN generator
    if args.dataset!="FFHQ":
        generator = Generator_(
            size = args.stylegan_size,
            style_dim = 512,
            n_mlp = 8,
            channel_multiplier = 2,
        )
    else:
        generator = Generator(
            size = 1024,
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
    align_model.prototypes = torch.Tensor(args.s_dict).to(args.device)
    align_model.to(args.device)

    return generator, align_model, s_dict, args


def run_global(generator, align_model, args, target, neutral):

    generated_images = []
    if args.random_latent:
        mean_latent = generator.mean_latent(4096)
        for i in range(50):
            latent_code_init_not_trunc = torch.randn(1, 512).cuda()
            with torch.no_grad():
                _, latent, _ = generator.forward([latent_code_init_not_trunc], return_latents=True,
                                            truncation=args.truncation, truncation_latent=mean_latent)
            torch.save(latent, f'latents/{args.dataset}/{i}.pt')
            img_orig, style_space, style_names, noise_constants = create_image_S(generator, latent)
            align_model.image_feature = align_model.encode_image(img_orig)
            generated_images.append(img_orig)
        generated = torch.cat(generated_images)
        save_image(generated, f'results/{args.dataset}.png', range=(-1, 1), normalize=True)
        exit()
    else:
        for i in range(5):
            latent = torch.load(f'latents/{args.dataset}/{i}.pt')
            latent.to(args.device)
            
            target_embedding = create_dt(target, model=align_model.model, neutral=neutral)
            align_model.text_feature = target_embedding
            group_name = os.path.join("results", args.dataset)
            print(generator.device, latent.device)
            img_orig, style_space, style_names, noise_constants = create_image_S(generator, latent, args)
            align_model.image_feature = align_model.encode_image(img_orig)
            generated_images.append(img_orig.detach().cpu())
            for _ in range(args.num_attempts):
                with torch.no_grad():
                    # Prepare model for evaluation
                    align_model.cross_modal_surgery()
                # StyleCLIP GlobalDirection 
                if args.method=="Baseline":
                    t = target_embedding.detach().cpu().numpy()
                    t = t/np.linalg.norm(t)
                else:
                    #! Ours (CosMos)
                    random_cores = align_model.diverse_text()
                    t, _ = align_model.postprocess(random_cores)
                img_gen, _, _ = manipulate_image(style_space, style_names, noise_constants, generator, latent, args, alpha=5, t=t, s_dict=s_dict, device=args.device)
                generated_images.append(img_gen.detach().cpu())
        generated_images = torch.cat(generated_images)
        os.makedirs(group_name, exist_ok=True)
        save_image(generated_images, f"{group_name}/{target}_{args.method}_diverse.png", normalize=True, range=(-1, 1))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configuration for styleCLIP Global Direction with our method')
    parser.add_argument('--method', type=str, default="Baseline", choices=["Baseline", "Random"], help='Use original styleCLIP global direction if Baseline')
    parser.add_argument('--num_attempts', type=int, default=5, help="Number of iterations for diversity measurement")
    parser.add_argument('--topk', type=int, default=25, help="Number of channels to modify")
    parser.add_argument('--beta', type=float, default=0.08, help="Number of channels to modify")
    parser.add_argument('--trg_lambda', type=float, default=0.5, help="weight for preserving the information of target")
    parser.add_argument('--temperature', type=float, default=1.0, help="Used for bernoulli")
    parser.add_argument("--truncation", type=float, default=0.7, help="used only for the initial latent vector, and only when a latent code path is"                                                                "not provided")
    parser.add_argument("--stylegan_size", type=int, default=512, help="StyleGAN resolution for AFHQ")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--random_latent", action='store_true', help='activate if you want to use random latent')
    parser.add_argument("--ir_se50_weights", type=str, default="../pretrained_models/model_ir_se50.pth")
    parser.add_argument("--segment_weights", type=str, default="../pretrained_models/79999_iter.pth")
    parser.add_argument("--dataset", type=str, default="FFHQ", choices=["FFHQ", "afhqdog", "afhqcat", "afhqwild"])
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    args.s_dict_path = f"./npy/{args.dataset}/fs3.npy"
    args.stylegan_weights = f"../pretrained_models/{args.dataset}.pt"
    
    generator, align_model, s_dict, args = prepare(args)

    args.targets = ['child', 'young']
    neutral = [""] * len(args.targets)

    ###########################################################

    for idx, target in enumerate(args.targets):
        args.neutral = neutral[idx]
        run_global(generator, align_model, args, target, args.neutral)