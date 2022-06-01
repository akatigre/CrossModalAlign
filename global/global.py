import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import tarfile
import argparse
import numpy as np
np.set_printoptions(suppress=True)

from utils.utils import *
from utils.global_dir_utils import create_dt, manipulate_image, manipulate_image_dir, create_image_S
# from utils.eval_utils import Text2Segment, maskImage
from model import CrossModalAlign
from models.stylegan2.models import Generator
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

def show(imgs, column_names, save_name, dpi=1800, suptitle=None):
    
    fig, axs = plt.subplots(nrows=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[i,0].imshow(np.asarray(img))
        axs[i,0].set_xlabel(column_names[i], fontsize=6)
        axs[i,0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=8)
    fig.subplots_adjust(bottom=0.1, hspace=0.35)
    plt.savefig(save_name, dpi=dpi)
    plt.cla() 


def prepare(args):
    if args.nsml: 
        import nsml
        with tarfile.open(os.path.join('..', nsml.DATASET_PATH, 'train','trained.tar.gz'), 'r') as f:
            f.extractall()

        print(os.listdir('./'))
        args.stylegan_weights = os.path.join("pretrained_models", "stylegan2-ffhq-config-f.pt")
        args.s_dict_path = os.path.join("global","npy", "ffhq", "fs3.npy")
        args.ir_se50_weights = os.path.join("pretrained_models", "model_ir_se50.pth")
        args.latents_path = os.path.join("pretrained_models", "test_faces.pt")

    # Load styleGAN generator
    generator = Generator(
        size = args.stylegan_size, # size of generated image
        style_dim = 512,
        n_mlp = 8,
        channel_multiplier = 2,
    )
    generator.load_state_dict(torch.load(args.stylegan_weights, map_location='cpu')['g_ema'])
    generator.eval()
    generator.to(args.device)

    # Load anchors
    s_dict = np.load(args.s_dict_path)
    args.s_dict = s_dict

    align_model = CrossModalAlign(args)
    align_model.prototypes = torch.Tensor(args.s_dict).to(args.device)
    align_model.to(args.device)

    return generator, align_model, args

def run_global(generator, align_model, args):

    test_latents = torch.load(args.latents_path, map_location='cpu')
    start_idx = 1
    latent = torch.Tensor(test_latents[start_idx][None]).to(args.device)
    
    # import lpips
    # lpips_alex = lpips.LPIPS(net='alex')
    # lpips_alex = lpips_alex.to(args.device)
    grids = []
    for target in args.targets:
        generated_images = []
        target_embedding = create_dt(target, model=align_model.model)
        align_model.text_feature = target_embedding
        
        # original Image from latent code (W+)
        img_orig, style_space, style_names, noise_constants = create_image_S(generator, latent)
        align_model.image_feature = align_model.encode_image(img_orig)
        generated_images.append(img_orig.detach().cpu().squeeze(0))
        
        # id_loss = AverageMeter()
        for _ in range(args.num_attempts):
            # StyleCLIP GlobalDirection 
            if args.method=="Baseline":
                t = target_embedding.detach().cpu().numpy()
                t = t/np.linalg.norm(t)
                img_gen, _, _ = manipulate_image(style_space, style_names, noise_constants, generator, latent, args, alpha=args.alpha, t=t, s_dict=args.s_dict, device=args.device)
            else:
                # Random Interpolation
                m_idxs, m_weights = align_model.cross_modal_surgery(fixed_weight=False)
                img_gen, _, _ = manipulate_image_dir(style_space, style_names, noise_constants, generator, latent, args, alpha=args.alpha, m_idxs=m_idxs, m_weights=m_weights, s_dict=args.s_dict, device=args.device)
            generated_images.append(img_gen.detach().cpu().squeeze(0))
            
            # Evaluation
            # with torch.no_grad():
            #     _id = align_model.evaluation(img_orig, img_gen, target)
            #     id_loss.update(_id)

        # with torch.no_grad(): 
        # # First image at generated image is original 
        #     segments = Text2Segment(target)
        # Activate Segment net before usage
        #     if args.num_attempts == 1 or len(segments) == 0: 
        #         lpips_value = 0.0
        #     else: 
        #         # PreProcess with Segmentation network 
        #         values, segmented_images = [], []
        #         for idx in range(1, args.num_attempts):
        #             # shape of [3, 1024, 1024] into [512, 512, 3]
        #             img = generated_images[idx]
        #             img = maskImage(img, Segment_net, args.device, segments, stride=1)
        #             if img is None:
        #                 continue
        #             segmented_images.append(img)

        #         N = len(segmented_images)

        #         for idx in range(1, N):
        #             tmp = lpips_alex(segmented_images[0], segmented_images[idx])
        #             values.append(tmp)
        #         lpips_value = sum(values) / (1.0* len(values))
        #         lpips_value = lpips_value[0][0][0][0].cpu().item()

        grid = make_grid(generated_images, nrow=args.num_attempts+1, normalize=True, value_range=(-1, 1))
        grids.append(grid)
    show(grids, column_names=args.targets, save_name=f'{args.dataset}.png', dpi=1800, suptitle=f"{args.method} latent: {start_idx} top: {args.topk} alpha: {args.alpha}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configuration for styleCLIP Global Direction with our method')
    parser.add_argument('--method', type=str, default="Baseline", choices=["Baseline", "Random"], help='Use original styleCLIP global direction if Baseline')
    parser.add_argument('--num_attempts', type=int, default=3, help="Number of iterations for diversity measurement")
    parser.add_argument('--topk', type=int, default=50, help="Number of channels to modify")
    parser.add_argument('--alpha', type=int, default=5, help="Manpulation strength")
    parser.add_argument('--num_test', type=int, default=1, help="Number of latents to test for debugging, if -1 then use all 100 images")
    parser.add_argument('--trg_lambda', type=float, default=0.5, help="weight for preserving the information of target")
    parser.add_argument('--temperature', type=float, default=1.0, help="Used for bernoulli")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument('--beta', type=float, default=0.15, help="Number of channels to modify")

    
    parser.add_argument("--nsml", action="store_true", help="run on the nsml server")
    parser.add_argument("--dataset", type=str, default="ffhq", choices=["ffhq", "afhqcat", "afhqdog", "church", 'car'])
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    args.stylegan_weights = f'../Pretrained/stylegan2/{args.dataset}.pt'
    args.s_dict_path = f'./dictionary/{args.dataset}/fs3.npy'
    args.latents_path = f'./latents/{args.dataset}/test_faces.pt'
    if args.dataset=='ffhq':
        args.stylegan_size = 1024
    elif args.dataset=='car':
        args.stylegan_size = 512
    elif args.dataset=='car':
        args.stylegan_size = 256

    generator, align_model, args = prepare(args)

    args.targets = ["man", 'man with long hair', 'Young', 'Old', 'Glasses', 'Smiling']
    
    args.neutral = ""
    run_global(generator, align_model, args)
