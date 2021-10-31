# python test_diversity.py --wandb --dir "global/results/Random/entangle"

import os
import glob
import torch
import wandb
import lpips
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

def pairwise_lpips(model, a, b, c):
    tmp = model.forward(a, b)
    tmp1 = model(a, c)
    tmp2 = model(b, c)
    return (tmp+tmp1+tmp2)*1.0 / 3.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="directory of the input image")
    parser.add_argument("--gpu", type=int, default=0, help="use specific gpu")
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()

    if args.wandb:
        tmp = args.dir.split('/')
        method = tmp[2]
        exp_name = f'diversity-{method}-{tmp[3]}'

        wandb.init(project="GlobalDirection", name=exp_name, group=method)
    
    
    assert os.path.exists(args.dir)
    name_list = glob.glob(os.path.join(args.dir, "*.png"))
    name_list = [f[:-6] for f in name_list]
    name_list = list(set(name_list))

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

 
    lpips_alex = lpips.LPIPS(net='alex')
    lpips_alex = lpips_alex.to(device)

    values = []
    toTorch = transforms.ToTensor()

    pbar = tqdm(name_list)

    with torch.no_grad():
        for name in pbar:
            file_name = os.path.join(args.dir, name)
            image1 = toTorch(Image.open(name+"-0.png", 'r')).to(device)
            image2 = toTorch(Image.open(name+"-1.png", 'r')).to(device)
            image3 = toTorch(Image.open(name+"-2.png", 'r')).to(device)
            
            value = pairwise_lpips(lpips_alex, image1, image2, image3)
            values.append(value.detach().cpu()[0][0][0][0])

            del image1
            del image2
            del image3
            pbar.set_description(f"Processing {name}")

    mean_value = sum(values) / len(values)
    print(f"Final LPIPS : {sum(values)}")
    if args.wandb:
        wandb.log({"lpips":mean_value})
        wandb.finish()

    


