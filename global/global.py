from genericpath import exists
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import wandb
import tensorflow as tf
import tarfile
import argparse
import numpy as np
np.set_printoptions(suppress=True)

from torchvision.utils import save_image, make_grid
from torchvision.transforms import ToPILImage
import wandb
from utils import *
from utils.utils import *
from model import CrossModalAlign
from models.stylegan2.models import Generator


def ffhq_style_semantic(channels):
    configs_ffhq = {
    'black hair' :      [(12, 479)],
    'blond hair ':      [(12, 479), (12, 266)],
    'grey hair ' :      [(11, 286)],
    'wavy hair'  :      [(6, 500), (8, 128), (5, 92), (6, 394), (6, 323)],
    'bangs'      :      [(3, 259), (6, 285), (5, 414), (6, 128), (9, 295), (6, 322), (6, 487), (6, 504)],
    'receding hairline':[(5, 414), (6, 322), (6, 497), (6, 504)],
    'smiling'    :      [(6, 501)],
    'lipstick'   :      [(15, 45)],
    'sideburns'  :      [(12, 237)],
    'goatee'     :      [(9, 421)],
    'earrings'   :      [(8, 81)],
    'glasses'    :      [(3, 288), (2, 175), (3, 120), (2, 97)],
    'wear suit'  :      [(9, 441), (8, 292), (11, 358), (6, 223)],
    'gender'     :      [(9, 6)]
    }
    style_channels = []
    for res, num_channels in channels.items():
        if res==4:
            style_channels.append(num_channels)
        else:
            style_channels.extend([num_channels]*2)
    
    mapped = {}
    for k, v in configs_ffhq.items():
        new_v = [sum(style_channels[:layer]) + ch for layer, ch in v]
        mapped[k] = new_v
    return mapped

def run_global(args, target, fs3, generator, device):

    test_latents = torch.load(args.latents_path, map_location='cpu')
    subset_latents = torch.Tensor(test_latents[0:args.num_test, :, :]).cpu() #len(test_latents)-args.num_test:len(test_latents)

    if args.method == "Baseline":
        exp_name = f"chNum{args.topk}"
    elif args.method == "Random":
        fs3 = disentangle_fs3(fs3)
        exp_name = f'target{target}-chNum{args.topk}-Weight{args.trg_lambda}'
    else:
        exp_name = None
        exit(-1)

    config = {
        "target": target,
        "method": args.method,
        "num channels": args.topk,
        "target weights": args.trg_lambda,
        "temperature": args.temperature
    }

    wandb.init(project="Global Direction", name=exp_name, group=args.method, config=config)

    align_model = CrossModalAlign(512, args)
    align_model.to(device)
    align_model.prototypes = torch.Tensor(fs3).to(device)
    
    target_embedding = GetTemplate(target, align_model.model).unsqueeze(0).float()
    align_model.text_feature = target_embedding

    for i, latent in enumerate(list(subset_latents)):
        latent = latent.unsqueeze(0).to(device)
        generated_images = []
        with torch.no_grad():
            style_space, style_names, noise_constants = encoder(generator, latent)
            img_orig = decoder(generator, style_space, latent, noise_constants)
            align_model.image_feature = align_model.encode_image(img_orig)
            
        generated_images.append(img_orig)
        manip_channels = set()
        id_loss, cs, ip, us = [AverageMeter() for _ in range(4)]
        for attmpt in range(args.num_attempts):
            with torch.no_grad():
                unwanted_mask, sc_mask = align_model.disentangle_diverse_text(lb=0.6, ub=0.3)
                image_positive_mask = align_model.extract_image_positive(unwanted_mask)
                align_model.unwanted_mask = [i for i in unwanted_mask if i not in image_positive_mask]
            # StyleCLIP GlobalDirection 
            if args.method=="Baseline":
                t = target_embedding.detach().cpu().numpy()
                t = t/np.linalg.norm(t)
            else:
            # Random Interpolation
                t = align_model.postprocess().detach().cpu().numpy()

            boundary_tmp2, num_c, dlatents, changed_channels = GetBoundary(fs3, t.squeeze(axis=0), args, style_space, style_names) # Move each channel by dStyle
            manip_channels.update(changed_channels)
            dlatents_loaded = [s.cpu().detach().numpy() for s in style_space]
            codes= MSCode(dlatents_loaded, boundary_tmp2, [5.0], device)
            img_gen = decoder(generator, codes, latent, noise_constants)
            generated_images.append(img_gen)
            with torch.no_grad():
                _id, _cs, _us, _ip = align_model.evaluation(img_orig, img_gen)
                id_loss.update(_id); cs.update(_cs); us.update(_us); ip.update(_ip)
            
        img_name =  f"img{len(subset_latents) - i}-{args.method}-{target}"
        img_dir = "results" if args.nsml else f"./results/{args.method}/{img_name}"
        os.makedirs(img_dir, exist_ok=True)
        generated_images = torch.cat(generated_images) # [1+num_attempts, 3, 1024, 1024]
        
        save_image(generated_images, f"{img_dir}/{img_name}.png", normalize=True, range=(-1, 1))
        wandb.log({
            f"{target}/Generated image": wandb.Image(generated_images, caption=img_name),
            f"{target}/core semantic": np.round(cs.avg, 3), 
            f"{target}/unwanted semantics": np.round(us.avg, 3), 
            f"{target}/source positive": np.round(ip.avg, 3),
            f"{target}/identity loss": id_loss.avg,
            f"{target}/channel idx": list(manip_channels)})

    wandb.finish() 

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configuration for styleCLIP Global Direction with our method')
    parser.add_argument('--method', type=str, default="Baseline", choices=["Baseline", "Random"], help='Use original styleCLIP global direction if Baseline')
    parser.add_argument('--targets', type=list)
    parser.add_argument('--num_attempts', type=int, default=5, help="Number of iterations for diversity measurement")
    parser.add_argument('--topk', type=int, default=50, help="Number of channels to modify")
    parser.add_argument('--beta', type=float, default=0., help="Threshold for style channel manipulation, topk used instead of beta")
    parser.add_argument('--trg_lambda', type=float, default=2.0, help="weight for preserving the information of target")
    parser.add_argument('--num_test', type=int, default=100, help="Number of latents to use for manipulation")
    parser.add_argument('--temperature', type=float, default=1.0, help="Used for bernoulli")
    parser.add_argument("--ir_se50_weights", type=str, default="../pretrained_models/model_ir_se50.pth")
    parser.add_argument("--stylegan_weights", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt")
    parser.add_argument("--latents_path", type=str, default="../pretrained_models/test_faces.pt")
    parser.add_argument("--fs3_path", type=str, default="./npy/ffhq/fs3.npy")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--nsml", action="store_true", help="run on the nsml server")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    generator = Generator(
            size = 1024, # size of generated image
            style_dim = 512,
            n_mlp = 8,
            channel_multiplier = 2
        )
    s_ch = ffhq_style_semantic(generator.channels)
    print(s_ch)
    if args.nsml: 
        import nsml
        with tarfile.open(os.path.join('..', nsml.DATASET_PATH, 'train','trained.tar.gz'), 'r') as f:
            f.extractall()

        print(os.listdir('./'))

        args.stylegan_weights = os.path.join("pretrained_models", "stylegan2-ffhq-config-f.pt")
        args.fs3_path = os.path.join("global","npy", "ffhq", "fs3.npy")
        args.ir_se50_weights = os.path.join("pretrained_models", "model_ir_se50.pth")
        args.latents_path = os.path.join("pretrained_models", "test_faces.pt")

    generator.load_state_dict(torch.load(args.stylegan_weights, map_location='cpu')['g_ema'])
    generator.eval()
    generator.to(device)

    wandb.login(key="5295808ee2ec2b1fef623a0b1838c5a5c55ae8d1")
    fs3 = np.load(args.fs3_path)
    args.targets = ["Arched eyebrows", "Bushy eyebrows", "Male", "Female", "Chubby", "Smiling", "Lipstick", "Eyeglasses", \
                    "Bangs", "Black hair", "Blond hair", "Straight hair", "Earrings", "Sidebunrs", "Goatee", "Receding hairline", "Grey hair", "Brown hair",\
                    "Wavy hair", "Wear suit", "Wear lipstick", "Double chin", "Hat", "Bags under eyes", "Big nose", "Big lips", "High cheekbones", "Young", "Old"]
    GLOBAL=["Male", "Female", "Young", "Old"]

    for target in args.targets:
        # if target not in GLOBAL: 
        #     channels = [50]
        # else:
        #     channels = [100, 200]
        channels = [50]
        for num_c in channels:
            if args.method=="Baseline":
                run_global(args, target, fs3, generator, device)
            elif args.method=="Random":
                for lmbd in [1.5]:
                    for temperature in [2.0]:
                        args.topk, args.trg_lambda, args.temperature = num_c, lmbd, temperature
                        print(f"Target Text: {target} Number of channels {num_c} Target lambda {lmbd} Temperature {temperature}")
                        run_global(args, target, fs3, generator, device)
