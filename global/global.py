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

from torchvision.utils import save_image
import wandb
from utils import *
from utils.utils import *
from model import CrossModalAlign
from models.stylegan2.models import Generator
from models.segment.model import BiSeNet


def run_global(args, target, fs3, generator, device):

    test_latents = torch.load(args.latents_path, map_location='cpu')
    test_indices = [1852, 2412, 226, 1070, 1603]#, 1276, 416, 1086, 200, 1033, 33, 598, 2726, 1977, 1945, 472, 629, 2144, 141, 1097, 1474, 149, 2050, 810, 831, 1385, 881, 1194, 2786, 2524, 2091, 2224, 113, 699, 1877, 1215, 382, 191, 1265, 611, 1685, 1034, 821, 1422, 1054, 2436, 1206, 1293, 2659, 973, 1198, 2242, 971, 1656, 2769, 1968, 2137, 2694, 352, 1701, 2561, 1657, 1229, 2666, 1416, 2250, 2635, 1976, 2585, 220, 1470, 1866, 2749, 2502, 1465, 714, 1687, 800, 421, 1185, 1868, 1808, 2227, 2145, 732, 2431, 2350, 323, 2706, 2239, 225, 1189, 1839, 2557, 650, 761, 892, 1430, 852, 1678]
    subset_latents = torch.Tensor(test_latents[test_indices, :, :]).cpu() 

    if args.method == "Baseline":
        exp_name = f"target{target}-chtopk{args.topk}"
    elif args.method == "Random":
        fs3 = disentangle_fs3(fs3)
        exp_name = f'target{target}-chNum{args.topk}-Weight{args.trg_lambda}-upper{args.ub}'
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

    wandb.init(project="Global Direction", name=exp_name, group=args.method+" without gamma weight", config=config)

    align_model = CrossModalAlign(512, args)
    align_model.to(device)
    align_model.prototypes = torch.Tensor(fs3).to(device)
    
    target_embedding = GetTemplate(target, align_model.model).unsqueeze(0).float()- GetTemplate(args.neutral, align_model.model).unsqueeze(0).float()
    align_model.text_feature = target_embedding

    import lpips
    lpips_alex = lpips.LPIPS(net='alex')
    lpips_alex = lpips_alex.to(device)

    Segment_net = BiSeNet(n_classes=19).to(device)
    ckpt = torch.load(args.segment_weights)
    Segment_net.load_state_dict(ckpt) 
    Segment_net.eval()

    for i, latent in enumerate(list(subset_latents)):
        latent = latent.unsqueeze(0).to(device)
        generated_images = []
        with torch.no_grad():
            style_space, style_names, noise_constants = encoder(generator, latent)
            img_orig = decoder(generator, style_space, latent, noise_constants)
            align_model.image_feature = align_model.encode_image(img_orig)
            
        generated_images.append(img_orig)
        manip_channels = set()
        id_loss, cs, ip, us, img_prop = [AverageMeter() for _ in range(5)]
        with torch.no_grad():
            align_model.cross_modal_surgery()

        for _ in range(args.num_attempts):
            # StyleCLIP GlobalDirection 
            if args.method=="Baseline":
                t = target_embedding.detach().cpu().numpy()
                t = t/np.linalg.norm(t)
            else:
            # Random Interpolation
                random_text_feature = l2norm(align_model.diverse_text())
                t, p = align_model.postprocess(random_text_feature)
                img_prop.update(p)
            boundary_tmp2, num_c, dlatents, changed_channels = GetBoundary(fs3, t.squeeze(axis=0), args, style_space, style_names) # Move each channel by dStyle
            manip_channels.update(changed_channels)
            dlatents_loaded = [s.cpu().detach().numpy() for s in style_space]
            codes= MSCode(dlatents_loaded, boundary_tmp2, [5.0], device)
            img_gen = decoder(generator, codes, latent, noise_constants)
            generated_images.append(img_gen)
            
            # Evaluation
            with torch.no_grad():
                _id, _cs, _us, _ip = align_model.evaluation(img_orig, img_gen)
                id_loss.update(_id); cs.update(_cs); us.update(_us); ip.update(_ip)

        with torch.no_grad(): 
        # First image at generated image is original 
            segments = Text2Segment(target)
            if args.num_attempts == 1 or len(segments) == 0: 
                lpips_value = 0.0
            else: 
                # PreProcess with Segmentation network 
                values, segmented_images = [], []
                for idx in range(1, args.num_attempts):
                    # shape of [3, 1024, 1024] into [512, 512, 3]
                    img = generated_images[idx]
                    img = maskImage(img, Segment_net, device, segments, stride=1)
                    if img is None:
                        continue
                    segmented_images.append(img)

                N = len(segmented_images)

                for idx in range(1, N):
                    tmp = lpips_alex(segmented_images[0], segmented_images[idx])
                    values.append(tmp)
                lpips_value = sum(values) / (1.0* len(values))

                lpips_value = lpips_value[0][0][0][0].cpu().item()


        img_name =  f"img{test_indices[i]}-{args.method}-{target}"
        img_dir = "results" if args.nsml else f"./results/{args.method}/{img_name}"
        # os.makedirs(img_dir, exist_ok=True)
        generated_images = torch.cat(generated_images) # [1+num_attempts, 3, 1024, 1024]
        # save_image(generated_images, f"{img_dir}/{img_name}.png", normalize=True, range=(-1, 1))

        wandb.log({
            f"{target}/Generated image": wandb.Image(generated_images, caption=img_name),
            f"core semantic": np.round(cs.avg, 3), 
            f"unwanted semantics": np.round(us.avg, 3), 
            f"source positive": np.round(ip.avg, 3),
            f"identity loss": id_loss.avg,
            f"channel idx": list(manip_channels),
            f"lpips": lpips_value,
            f"image proportion": img_prop.avg,
            }
            )

    wandb.finish()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configuration for styleCLIP Global Direction with our method')
    parser.add_argument('--method', type=str, default="Baseline", choices=["Baseline", "Random"], help='Use original styleCLIP global direction if Baseline')
    parser.add_argument('--num_attempts', type=int, default=5, help="Number of iterations for diversity measurement")
    parser.add_argument('--trg_lambda', type=float, default=0.1, help="weight for preserving the information of target")
    parser.add_argument('--temperature', type=float, default=2.0, help="Used for bernoulli")
    parser.add_argument("--ir_se50_weights", type=str, default="../pretrained_models/model_ir_se50.pth")
    parser.add_argument("--stylegan_weights", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt")
    parser.add_argument("--segment_weights", type=str, default="./79999_iter.pth")
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
            channel_multiplier = 2,
        )
        
    # TODO: Upload afhq_cat/fs3.npy & afhq_dog/fs3.npy on nsml
    if args.nsml: 
        import nsml
        with tarfile.open(os.path.join('..', nsml.DATASET_PATH, 'train','trained.tar.gz'), 'r') as f:
            f.extractall()

        print(os.listdir('./'))
        # TODO: Change styleGAN pretrained model & ffhq -> afhq
        args.stylegan_weights = os.path.join("pretrained_models", "stylegan2-ffhq-config-f.pt")
        args.fs3_path = os.path.join("global","npy", "ffhq", "fs3.npy")
        args.ir_se50_weights = os.path.join("pretrained_models", "model_ir_se50.pth")
        # TODO: Randomly generate 200 latents of afhq_dog / afhq_cat respectively -> use as test_faces
        args.latents_path = os.path.join("pretrained_models", "test_faces.pt")


    generator.load_state_dict(torch.load(args.stylegan_weights, map_location='cpu')['g_ema'])
    generator.eval()
    generator.to(device)


    # TODO: Write down the wandb API key below
    wandb.login(key="5295808ee2ec2b1fef623a0b1838c5a5c55ae8d1")
    fs3 = np.load(args.fs3_path)

    # Text set A: 수민
    # args.targets = ["Arched eyebrows", "Bushy eyebrows", "Male", "Female", "Chubby", "Smiling", "Lipstick", "Eyeglasses", \
                    # "Bangs", "Black hair", "Blond hair", "Straight hair", "Earrings", "Sidebunrs"]
    # neutral = ["Eye", "Eye", "Female", "Male", "Face", "Face", "Face", "Face", "Hair", "Hair", "Hair", "Hair", "Face", "Face"]
    # upper_bound = [0.1, 0.1, 0.6, 0.2, 0.6, 0.4, 0.1, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6] 
    
    # Text set B: 윤전 
    # args.targets = ["Goatee", "Receding hairline", "Grey hair", "Brown hair",
    #                 "Wavy hair", "Wear suit", "Double chin", "Hat", "Bags under eyes",
    #                 "Big nose", "Big lips", "High cheekbones", "Young", "Old"]
    # neutral = ["Face", "Hair", "Hair", "Hair", "Hair", "", "Face", "", "eye", "Nose", "Lips", "Face", "Old", "Young"]
    # upper_bound = [0.6, 0.4, 0.6, 0.6, 0.6, 0.4, 0.6, 0.6, 0.6, 0.2, 0.2, 0.2, 0.6, 0.6]



    # Full Text set
    args.targets = ["Arched eyebrows", "Bushy eyebrows", "Male", "Female", "Chubby", "Smiling", "Lipstick", "Eyeglasses", \
                    "Bangs", "Black hair", "Blond hair", "Straight hair", "Earrings", "Sidebunrs", "Goatee", "Receding hairline", "Grey hair", "Brown hair",\
                    "Wavy hair", "Wear suit", "Double chin", "Hat", "Bags under eyes", "Big nose", "Big lips", "High cheekbones", "Young", "Old"]
    neutral = ["Eye", "Eye", "Female", "Male", "Face", "Face", "Face", "Face", "Hair", "Hair", "Hair", "Hair", "Face", "Face", "Face", "Hair", "Hair", "Hair", "Hair", "", "Face", "", "eye", "Nose", "Lips", "Face", "Old", "Young"]
    upper_bound = [0.1, 0.1, 0.6, 0.2, 0.6, 0.4, 0.1, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.6, 0.6, 0.6, 0.4, 0.6, 0.6, 0.6, 0.2, 0.2, 0.2, 0.6, 0.6] 

    channels = [25, 50, 100]
    assert len(upper_bound)==len(args.targets)
    for idx, target in enumerate(args.targets):
        for num_c in channels:
            args.topk = num_c
            args.neutral = neutral[idx]
            args.ub = upper_bound[idx]
            run_global(args, target, fs3, generator, device)
