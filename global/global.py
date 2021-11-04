import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from utils import *
import clip
import argparse
from models.stylegan2.models import Generator
import numpy as np
from torch.nn import functional as F
from utils.utils import *
from torchvision.utils import save_image
from criteria.id_loss import IDLoss
import wandb
from text_model import RandomInterpolation
np.set_printoptions(suppress=True)
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
    skip = G.to_rgb1(out, latent[:, 0])

    i = 2; j = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        out = conv_warper(conv1, out, style_space[i], noise=noise1)
        out = conv_warper(conv2, out, style_space[i+1], noise=noise2)
        skip = to_rgb(out,  latent[:, j + 2], skip)

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
    parser.add_argument('--method', type=str, default="baseline", choices=["Baseline", "Random"], help='Use original styleCLIP global direction if Baseline')
    parser.add_argument('--num_attempts', type=int, default=1, help="Number of iterations for diversity measurement")
    parser.add_argument('--beta', type=float, default=0.)
    parser.add_argument('--topk', type=int, default=50, help="Number of channels to modify")
    parser.add_argument('--trg_lambda', type=float, default=2.0, help="weight for preserving the information of target")
    parser.add_argument('--num_test', type=int, default=1, help="Number of latents to use for manipulation")
    parser.add_argument('--temperature', type=float, default=1.0, help="Used for bernoulli")
    parser.add_argument('--easy', action="store_true", default=False)
    parser.add_argument("--ir_se50_weights", type=str, default="../models/model_ir_se50.pth")
    parser.add_argument("--stylegan_weights", type=str, default="../models/stylegan2-ffhq-config-f.pt")
    args = parser.parse_args()
    device = "cuda:0"
    config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}
    
    generator = Generator(
            size = 1024, # size of generated image
            style_dim = config["latent"],
            n_mlp = config["n_mlp"],
            channel_multiplier = config["channel_multiplier"]
        )
    
    generator.load_state_dict(torch.load(args.stylegan_weights, map_location='cpu')['g_ema'])
    generator.eval()
    generator.cuda()

    idloss = IDLoss(args).cuda()
    model, preprocess = clip.load("ViT-B/32", device = device)
    fs3 = np.load("./npy/ffhq/fs3.npy") # 6048, 512
 
    test_latents = torch.load("../models/test_faces.pt", map_location='cpu')
    # test_latents = torch.Tensor(test_latents[len(test_latents)-args.num_test:, :]).cpu()
    test_latents = torch.Tensor(test_latents[25, :, :]).cpu()

    if args.method == "Baseline":
        exp_name = f"method{args.method}"
    else:
        newfs3 = disentangle_fs3(fs3)
        np.save("pca5fs3.npy", newfs3)
        print(f"Original: {uniform_loss(fs3)} After Disentangle: {uniform_loss(newfs3)}")
        fs3 = newfs3
        exp_name = f'method{args.method}-chNum{args.topk}-src{args.trg_lambda}'
    if args.easy:
        exp_name += "easy"
        descriptions = ["asian", "grey hair", "red hair", "wears earrings"]
    else:
        exp_name += 'hard'
        hard_test = []
        descriptions = hard_test

    wandb.init(project="GlobalDirection", name=exp_name, group=args.method, config = config)
        
    for target in descriptions:
        # for i, latent in enumerate(list(test_latents)):
        latent = test_latents.unsqueeze(0).cuda()
        i = "test"
        
        for attmpt in range(args.num_attempts):
            with torch.no_grad():
                style_space, style_names, noise_constants = encoder(generator, latent)
                img_orig = decoder(generator, style_space, latent, noise_constants)
                target_embedding = GetDt(target, model)
            args.description = target_embedding.unsqueeze(0).float()
            text_model = RandomInterpolation(512, model, preprocess, device, img_orig, args, fs3)
            text_model.cuda()
            # StyleCLIP GlobalDirection 
            if args.method=="Baseline":
                target_embedding = target_embedding.detach().cpu().numpy()
                target_embedding = target_embedding/np.linalg.norm(target_embedding)
            else:
            # Random Interpolation
                image_manifold, gamma = text_model()
                disen_text = text_model.text_feature
                #! Use target_embedding instead of disen_text
                text_star = l2norm(args.trg_lambda * gamma * disen_text + image_manifold)
                target_embedding = text_star.squeeze(0).detach().cpu().numpy()

            boundary_tmp2, num_c, dlatents = GetBoundary(fs3, target_embedding, args, style_space, style_names) # Move each channel by dStyle
            dlatents_loaded = [s.cpu().detach().numpy() for s in style_space]
            codes= MSCode(dlatents_loaded, boundary_tmp2, [5.0], device)
            img_gen = decoder(generator, codes, latent, noise_constants)
            
            # Save each image in proper directory
            img_name =  f"{i}-{args.method}-{target}-{attmpt}"
            imgs = torch.cat([img_orig, img_gen])
            # img_dir = f"results/{args.method}/{entangle}/"
            # if not os.path.exists(img_dir):
            #     os.mkdir(img_dir)

            save_image(imgs, f"{img_name}.png", normalize=True, range=(-1, 1))
            with torch.no_grad():
                identity = idloss(img_orig, img_gen)[0]
                new_image_feature = text_model.encode_image(img_gen)
                cs, us, ip = text_model.evaluation(new_image_feature)

            wandb.log({
                "Generated image": wandb.Image(imgs, caption=img_name),
                "core semantic": np.round(cs, 3), 
                "unwanted semantics": np.round(us, 3), 
                "source positive": np.round(ip, 3),
                "identity loss": identity,
                "changed channels": num_c})
wandb.finish() 
