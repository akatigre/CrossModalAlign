import os
import clip
import wandb
import torch
import argparse
import numpy as np
from utils import *
from torch.nn import functional as F
from torchvision.utils import save_image

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from criteria.id_loss import IDLoss
from stylegan2.model import Generator
from text_model import RandomInterpolation


def conv_warper(layer, input, style, noise):
    # the conv should change
    conv = layer.conv
    batch, in_channel, height, width = input.shape
    style = style.view(batch, 1, in_channel, 1, 1)
    weight = conv.scale * conv.weight * style

    if conv.demodulate:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

    weight = weight.view(
        batch * conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
    )

    if conv.upsample: # up==2
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, conv.out_channel, conv.kernel_size, conv.kernel_size
        )
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        out = conv.blur(out)

    elif conv.downsample:
        input = conv.blur(input)
        _, _, height, width = input.shape
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    else:
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=conv.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
    out = layer.noise(out, noise=noise)
    out = layer.activate(out)
    
    return out

def decoder(G, style_space, latent, noise):
    """
    Returns array of generated image from manipulated style space
    """

    out = G.input(latent)

    out = conv_warper(G.conv1, out, style_space[0], noise[0])
    skip, _ = G.to_rgb1(out, latent[:, 0])

    i = 2; j = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        out = conv_warper(conv1, out, style_space[i], noise=noise1)
        out = conv_warper(conv2, out, style_space[i+1], noise=noise2)
        skip, _ = to_rgb(out, latent[:, j + 2], skip)

        i += 3; j += 2

    image = skip

    return image

def encoder(G, latent): 
    noise_constants = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]
    style_space = []
    style_names = []
    # rgb_style_space = []
    style_space.append(G.conv1.conv.modulation(latent[:, 0]))
    res=4
    style_names.append(f"b{res}/conv1")
    style_space.append(G.to_rgbs[0].conv.modulation(latent[:, 0]))
    style_names.append(f"b{res}/torgb")
    i = 1; j=3
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise_constants[1::2], noise_constants[2::2], G.to_rgbs
    ):
        res=2**j
        style_space.append(conv1.conv.modulation(latent[:, i]))
        style_names.append(f"b{res}/conv1")
        style_space.append(conv2.conv.modulation(latent[:, i+1]))
        style_names.append(f"b{res}/conv2")
        style_space.append(to_rgb.conv.modulation(latent[:, i + 2]))
        style_names.append(f"b{res}/torgb")
        i += 2; j += 1
        
    return style_space, style_names, noise_constants


    



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configuration for styleCLIP')
    parser.add_argument('--method', type=str, default="Baseline", choices=["Baseline", "Random"], help='Use original styleCLIP global direction')
    parser.add_argument('--disentangle_fs3', default="T", choices = ["T", "F"], help='Use disentangling fs3')
    parser.add_argument('--q', type=float, help='Quantile for selecting the threshold')
    parser.add_argument('--num_test', type=int, default=50)
    parser.add_argument("--ir_se50_weights", type=str, default="../pretrained_models/model_ir_se50.pth")
    parser.add_argument("--num_attempts", default=3, type=int)
    args = parser.parse_args()
    device = "cuda:0"
    config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}

    generator = Generator(
            size = 1024, # size of generated image
            style_dim = config["latent"],
            n_mlp = config["n_mlp"],
            channel_multiplier = config["channel_multiplier"]
        )
    
    # Load pretrained stylegan2 pytorch
    generator.load_state_dict(torch.load("../pretrained_models/stylegan2-ffhq-config-f.pt", map_location='cpu')['g_ema'])
    generator.eval()
    generator.cuda()
    idloss = IDLoss(args).cuda()
    model, preprocess = clip.load("ViT-B/32", device = device)
    fs3 = np.load('./npy/ffhq/fs3.npy') # 6048, 512
    np.set_printoptions(suppress=True) 
   
    test_latents = torch.load("../mapper/test_faces.pt", map_location='cpu')
    test_latents = torch.Tensor(test_latents[:args.num_test, :]).cpu()
    exp_name = f'method{args.method}-q{args.q}-disentangle{args.disentangle_fs3}'

    tags = []
    if args.disentangle_fs3=="T":
        tags.append("disentangle fs3")
    tags.append(f"quantile {args.q}")
    wandb.init(project="GlobalDirection", name=exp_name, group=args.method, tags=tags)
    
    for target in ["grey hair", "he wears lipstick", "she is grumpy", "asian is smiling"]:
        for i, latent in enumerate(test_latents):
            for attmpt in range(len(args.num_attempts)):
                with torch.no_grad():
                    latent = latent.unsqueeze(0).cuda()
                    style_space, style_names, noise_constants = encoder(generator, latent)
                    img_orig = decoder(generator, style_space, latent, noise_constants)
                    target_embedding = GetDt(target, model)
                    args.description = target_embedding.unsqueeze(0).float()
                    text_model = RandomInterpolation(512, model, preprocess, device, img_orig, args)
                    text_model.cuda()

                # StyleCLIP GlobalDirection 
                if args.method=="Baseline":
                    target_embedding = target_embedding.detach().cpu().numpy()
                    target_embedding = target_embedding/np.linalg.norm(target_embedding)
                else:
                # Random Interpolation
                    text_feature = text_model.text_feature
                    image_manifold, gamma = text_model()
                    text_star = l2norm(2 * gamma * text_feature + image_manifold)
                    target_embedding = text_star.squeeze(0).detach().cpu().numpy()

                if args.disentangle_fs3=="T":
                    # Disentangle style channels
                    for i in range(len(fs3)):
                        fs3 = torch.Tensor(fs3)
                        t = fs3[i, :]
                        sim = torch.Tensor(fs3 @ t.T)
                        _, core = torch.topk(sim, k=1)
                        _, us = torch.topk(sim, k=3)
                        us_fs3 = torch.stack([fs3[i] for i in us if i not in core])
                        weights = torch.stack([sim[i] for i in us if i not in core])
                        int_sem = l2norm(torch.mm(weights.unsqueeze(0), us_fs3))
                        fs3[i, :] = projection(basis=t, target=int_sem)
                        fs3 = fs3.numpy()
                if args.method == "Baseline": 
                    boundary_tmp2, c, dlatents = GetBoundary(fs3, target_embedding, args.q, style_space, style_names) # Move each channel by dStyle
                else:
                    boundary_tmp2, c, dlatents = GetBoundary(fs3, target_embedding, 50, style_space, style_names) # Move each channel by dStyle
                dlatents_loaded = [s.cpu().detach().numpy() for s in style_space]
                codes= MSCode(dlatents_loaded, boundary_tmp2, [5.0], device)
                img_gen = decoder(generator, codes, latent, noise_constants)
                
                # Save each image in proper directory
                entangle = "disentangle" if args.disentangle_fs3=="T" else "entangle"
                img_name =  f"{i}-{target}-{attmpt}"
                imgs = torch.cat([img_orig, img_gen])
                img_dir = f"results/{args.method}/{entangle}/"
                os.makedirs(img_dir, exist_ok=True)
                save_image(imgs, os.path.join(img_dir, f"{img_name}.png"), normalize=True, range=(-1, 1))
                with torch.no_grad():
                    identity = idloss(img_orig, img_gen)[0]
                    new_image_feature = text_model.encode_image(img_gen)
                    cs, us, ip = text_model.evaluation(new_image_feature)
                wandb.log({"Generated image": wandb.Image(imgs, caption=img_name), 
                            "core semantic": np.round(cs, 3), 
                            "unwanted semantics": np.round(us, 3), 
                            "source positive": np.round(ip, 3),
                            "identity loss": identity, 
                            "changed channels": c})
    wandb.finish() 
