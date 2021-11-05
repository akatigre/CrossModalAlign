from genericpath import exists
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import tensorflow as tf
import tarfile
import argparse
import numpy as np
np.set_printoptions(suppress=True)

from torchvision.utils import save_image
from torchvision.transforms import ToPILImage

from utils import *
from utils.utils import *
from model import CrossModalAlign
from models.stylegan2.models import Generator

descriptions = ["asian", "grey hair", "red hair", "wears earrings", "smiling", "african", "terrorist"]

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configuration for styleCLIP Global Direction with our method')
    parser.add_argument('--method', type=str, default="Baseline", choices=["Baseline", "Random"], help='Use original styleCLIP global direction if Baseline')
    parser.add_argument('--target', type=str, help="Target text to manipulate the source image")
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
    parser.add_argument("--wandb", type=bool, default=True, help="Whether to activate the wandb")
    parser.add_argument("--nsml", action="store_true", help="run on the nsml server")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    generator = Generator(
            size = 1024, # size of generated image
            style_dim = 512,
            n_mlp = 8,
            channel_multiplier = 2
        )

    if args.nsml: 
        import nsml
        # if os.path.exists(os.path.join('..', nsml.DATASET_PATH, 'train')):
        #     print("exist 1")
        #     print(os.listdir(os.path.join('..', nsml.DATASET_PATH, 'train')))
        # print(os.listdir(os.path.join('..', nsml.DATASET_PATH, 'train','trained.tar.gz')))
    
        with tarfile.open(os.path.join('..', nsml.DATASET_PATH, 'train','trained.tar.gz'), 'r') as f:
            f.extractall()

        print(os.listdir('./'))

        args.stylegan_weights = os.path.join("pretrained_models", "stylegan2-ffhq-config-f.pt")
        args.fs3_path = os.path.join("global","npy", "ffhq", "fs3.npy")
        args.ir_se50_weights = os.path.join("pretrained_models", "model_ir_se50.pth")
        args.latents_path = os.path.join("pretrained_models", "test_faces.pt")
        args.wandb = False

    generator.load_state_dict(torch.load(args.stylegan_weights, map_location='cpu')['g_ema'])
    generator.eval()
    generator.to(device)

    fs3 = np.load(args.fs3_path)
 
    test_latents = torch.load(args.latents_path, map_location='cpu')
    subset_latents = torch.Tensor(test_latents[len(test_latents)-args.num_test:len(test_latents), :, :]).cpu()

    if args.method == "Baseline":
        exp_name = f"method{args.method}-chNum{args.topk}"
    elif args.method == "Random":
        fs3 = disentangle_fs3(fs3)
        exp_name = f'method{args.method}-chNum{args.topk}-targetWeight{args.trg_lambda}'
    else:
        exp_name = None
        exit(-1)

    config = {
        "target": args.target,
        "method": args.method,
        "num channels": args.topk,
        "target weights": args.trg_lambda,
    }

    if args.wandb:
        import wandb
        wandb.init(project="GlobalDirection", name=exp_name, group=args.method, config=config)
    align_model = CrossModalAlign(512, args)
    align_model.to(device)
    align_model.prototypes = torch.Tensor(fs3).to(device)
    
    target_embedding = GetTemplate(args.target, align_model.model).unsqueeze(0).float()
    align_model.text_feature = target_embedding
    unwanted_mask, sc_mask = align_model.disentangle_text(lb=0.6, ub=0.3)

    for i, latent in enumerate(list(subset_latents)):
        latent = latent.unsqueeze(0).to(device)
        for attmpt in range(args.num_attempts):
            with torch.no_grad():
                style_space, style_names, noise_constants = encoder(generator, latent)
                img_orig = decoder(generator, style_space, latent, noise_constants)

                align_model.image_feature = align_model.encode_image(img_orig)
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
            
            img_name =  f"img{len(subset_latents) - i}-{args.method}-{args.target}-{attmpt}"
            img_dir = f"./results/{args.method}/{exp_name}"

            os.makedirs(img_dir, exist_ok=True)

            imgs = torch.cat([img_orig, img_gen]) #shape : [2, 3, 1024, 1024]

             
            if not args.nsml:
                save_image(imgs, f"{img_dir}{img_name}.png", normalize=True, range=(-1, 1))
                # nsml 에서 내부 이미지 변화시켜야 함 

            
            with torch.no_grad():
                identity, cs, us, ip = align_model.evaluation(img_orig, img_gen)
            
            if args.wandb:
                logs = {
                    "Generated image": wandb.Image(imgs, caption=img_name),
                    "core semantic": np.round(cs, 3), 
                    "unwanted semantics": np.round(us, 3), 
                    "source positive": np.round(ip, 3),
                    "identity loss": identity,
                    "changed channels": num_c}
                wandb.log(**logs)
            if args.nsml:
                
                logs = {
                    "core semantic": float(np.round(cs, 3)[0]), 
                    "unwanted semantics": float(np.round(us, 3)[0]), 
                    "source positive": float(np.round(ip, 3)[0]),
                    "identity loss": identity.item(),
                    "changed channels": num_c}
                nsml.report(**logs, scope=locals())
                
    if args.wandb:
        wandb.finish() 
