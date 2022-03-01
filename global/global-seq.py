import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import tarfile
import argparse
import numpy as np
np.set_printoptions(suppress=True)

from utils.utils import *
from utils.global_dir_utils import create_dt, manipulate_image, create_image_S
# from utils.eval_utils import Text2Segment, maskImage
from model import CrossModalAlign
from models.stylegan2.models import Generator
from torchvision.utils import save_image

def prepare(args):
    # Load styleGAN generator
    generator = Generator(
        size = 1024, # size of generated image
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
        args.s_dict = project_away_pc(s_dict, k=5)
    elif args.method=="Baseline":
        args.s_dict = s_dict

    align_model = CrossModalAlign(args)
    align_model.prototypes = torch.Tensor(args.s_dict).to(args.device)
    align_model.to(args.device)

    return generator, align_model, args

def run_global(generator, align_model, args, targets, neutrals):
    test_latents = torch.load(args.latents_path, map_location='cpu')
    start_idx = 70
    subset_latents = torch.Tensor(test_latents[start_idx:start_idx+args.num_test]).cpu()
    img_dir = f"Composition-{args.method}-{args.dataset}"
    os.makedirs(img_dir, exist_ok=True)

    for i, latent in enumerate(list(subset_latents)):
        latent = latent.unsqueeze(0).to(args.device)
        generated_images = []
        # original Image from latent code (W+)
        img_orig, style_space, style_names, noise_constants = create_image_S(generator, latent)
        align_model.image_feature = align_model.encode_image(img_orig)
        generated_images.append(img_orig)

        id_loss = AverageMeter()
        
        for idx, target in enumerate(targets):
            # StyleCLIP GlobalDirection 
            target_embedding = create_dt(target, model=align_model.model, neutral=neutrals[idx])
            align_model.text_feature = target_embedding

            if args.method=="Baseline":
                t = target_embedding.detach().cpu().numpy()
                t = t/np.linalg.norm(t)
            else:
                # Random Interpolation
                t = align_model.cross_modal_surgery().detach().cpu().numpy()
            if idx >= 1:
                t = t+prev_text
                t = t/ np.linalg.norm(t)
            img_gen, _, _ = manipulate_image(style_space, style_names, noise_constants, generator, latent, args, alpha=5, t=t, s_dict=args.s_dict, device=args.device)
            
            prev_text = t
            generated_images.append(img_gen)
            align_model.image_feature = align_model.encode_image(img_gen)
            
            # Evaluation
            with torch.no_grad():
                _id = align_model.evaluation(img_orig, img_gen, target)
                id_loss.update(_id)
                
        img_name = f"img-{args.method}-{start_idx}-{target}"
        generated_images = torch.cat(generated_images) # [1+num_attempts, 3, 1024, 1024]
        save_image(generated_images, f"{img_dir}/{img_name}.png", normalize=True, range=(-1, 1))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configuration for styleCLIP Global Direction with our method')
    parser.add_argument('--method', type=str, default="Baseline", choices=["Baseline", "Random"], help='Use original styleCLIP global direction if Baseline')
    parser.add_argument('--topk', type=int, default=50, help="Number of channels to modify", choices=[25, 50, 100])
    parser.add_argument('--num_test', type=int, default=1, help="Number of latents to test for debugging, if -1 then use all 100 images")
    parser.add_argument('--trg_lambda', type=float, default=0.5, help="weight for preserving the information of target")
    parser.add_argument('--temperature', type=float, default=1.0, help="Used for bernoulli")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")

    ### MODEL WEIGHT
    parser.add_argument("--ir_se50_weights", type=str, default="../pretrained_models/model_ir_se50.pth")
    parser.add_argument("--stylegan_weights", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt")
    parser.add_argument("--segment_weights", type=str, default="../pretrained_models/79999_iter.pth")
    parser.add_argument("--latents_path", type=str, default="../pretrained_models/train_faces.pt")
    parser.add_argument("--s_dict_path", type=str, default="./npy/ffhq/fs3.npy")
    
    parser.add_argument("--nsml", action="store_true", help="run on the nsml server")
    parser.add_argument("--dataset", type=str, default="FFHQ", choices=["FFHQ", "AFHQ"])
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    
    generator, align_model, args = prepare(args)

    targets = ['young','purple hair', 'curly hair']
    neutral = [""] * len(targets)

    run_global(generator, align_model, args, targets, neutral)
