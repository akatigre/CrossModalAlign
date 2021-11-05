import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from utils import *
import argparse
from models.stylegan2.models import Generator
import numpy as np
np.set_printoptions(suppress=True)
from utils.utils import *
from torchvision.utils import save_image
import wandb
from model import CrossModalAlign


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configuration for styleCLIP Global Direction with our method')
    parser.add_argument('--method', type=str, default="Baseline", choices=["Baseline", "Random"], help='Use original styleCLIP global direction if Baseline')
    parser.add_argument('--topk', type=int, default=50, help="Number of channels to modify")
    parser.add_argument('--beta', type=float, default=0., help="Threshold for style channel manipulation, topk used instead of beta")
    parser.add_argument('--trg_lambda', type=float, default=2.0, help="weight for preserving the information of target")
    parser.add_argument('--num_test', type=int, default=2, help="Index of latent to use")
    parser.add_argument('--temperature', type=float, default=1.0, help="Used for bernoulli")
    parser.add_argument("--ir_se50_weights", type=str, default="../pretrained_models/model_ir_se50.pth")
    parser.add_argument("--stylegan_weights", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt")
    parser.add_argument("--latents_path", type=str, default="../pretrained_models/test_faces.pt")
    parser.add_argument("--fs3_path", type=str, default="./npy/ffhq/fs3.npy")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--wandb", type=bool, default=True, help="Whether to activate the wandb")
    args = parser.parse_args()
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
 
    test_latents = torch.load(args.latents_path, map_location='cpu')
    subset_latents = torch.Tensor(test_latents[args.num_test:args.num_test+1, :, :]).cpu()

    descriptions = ["Male", "Without Beard"]

    if args.method == "Baseline":
        exp_name = f"sequential-method{args.method}-chNum{args.topk}"
    elif args.method == "Random":
        fs3 = disentangle_fs3(fs3)
        exp_name = f'sequential-method{args.method}-chNum{args.topk}-targetWeight{args.trg_lambda}'
    else:
        exp_name = None
        NotImplementedError()

    config = {
        "description_list": descriptions,
        "method": args.method,
        "num channels": args.topk,
        "target weights": args.trg_lambda,
    }

    if args.wandb:
        wandb.init(project="GlobalDirection", name=exp_name, group=args.method, config=config)
    align_model = CrossModalAlign(512, args)
    align_model.to(device)
    align_model.prototypes = torch.Tensor(fs3).to(device)
    
    latent = subset_latents[0].unsqueeze(0).to(device)
    imgs = []

    for idx, target in enumerate(descriptions):
        target_embedding = GetTemplate(target, align_model.model).unsqueeze(0).float()
        align_model.text_feature = target_embedding
        unwanted_mask, sc_mask = align_model.disentangle_text(lb=0.6, ub=0.3)

        with torch.no_grad():
            if idx == 0:
                style_space, style_names, noise_constants = encoder(generator, latent)
                img_orig = decoder(generator, style_space, latent, noise_constants)
                imgs.append(img_orig)

            align_model.ßimage_feature = align_model.encode_image(img_orig)
            image_positive_mask = align_model.extract_image_positive(unwanted_mask)
            align_model.unwanted_mask = [i for i in unwanted_mask if i not in image_positive_mask]

        # StyleCLIP GlobalDirection 
        if args.method=="Baseline":
            t = target_embedding.detach().cpu().numpy()
            t = t/np.linalg.norm(t)
        else:
        # Random Interpolation
            t = align_model.postprocess().detach().cpu().numpy()

        boundary_tmp2, num_c, dlatents = GetBoundary(fs3, t.squeeze(axis=0), args, style_space, style_names) # Move each channel by dStyle
        dlatents_loaded = [s.cpu().detach().numpy() for s in style_space]
        codes= MSCode(dlatents_loaded, boundary_tmp2, [5.0], device)
        img_gen = decoder(generator, codes, latent, noise_constants)
        imgs.append(img_gen)
        style_space = codes
        img_orig = img_gen 
    
    with torch.no_grad():
        img_name = f"Sequential-{args.num_test}-{args.method}-{descriptions[0]}"
        img_dir = f"results/sequential/{args.method}/{exp_name}"

        os.makedirs(img_dir, exist_ok=True)
        imgs = torch.cat(imgs)
        save_image(imgs, f"{img_dir}/{img_name}.png", normalize=True, range=(-1, 1))
    if args.wandb:
        wandb.log({
                "Generated image": wandb.Image(imgs, caption=img_name)})
        wandb.finish() 
