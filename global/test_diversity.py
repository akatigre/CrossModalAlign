# python test_diversity.py --wandb --dir "global/results/Random/entangle"
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import glob
import torch
import wandb
import lpips
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import crop

from criteria.id_loss import IDLoss

def pairwise_lpips(model, imgs):
    if len(imgs) == 1: 
        return -1
    if len(imgs) == 2:
        return model.forward(imgs[0], imgs[1])
    
    tmp = model.forward(imgs[0], imgs[1])
    tmp1 = model(imgs[0], imgs[1])
    tmp2 = model(imgs[1], imgs[2])
    return (tmp+tmp1+tmp2)*1.0 / 3.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="directory of the input image")
    parser.add_argument("--gpu", type=int, default=0, help="use specific gpu")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ir_se50_weights", type=str, default="../pretrained_models/model_ir_se50.pth")
    parser.add_argument("--conditioned", type=bool, default=True)

    args = parser.parse_args()

    if args.wandb:
        tmp = args.dir.split('/')
        method = tmp[2]
        exp_name = f'diversity-{method}-{tmp[3]}'

        wandb.init(project="Global Direction Diversity Checking", name=exp_name, group=method)
    
    
    assert os.path.exists(args.dir)
    name_list = glob.glob(os.path.join(args.dir, "*.png"))
    name_list = [f[:-6] for f in name_list]
    name_list = list(set(name_list))

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    criteria = IDLoss(args).to(device)

    lpips_alex = lpips.LPIPS(net='alex')
    lpips_alex = lpips_alex.to(device)

    values = []
    toTorch = transforms.ToTensor()

    pbar = tqdm(name_list)

    image1 = Image.open(f"{name_list[0]}-0.png", 'r')
    size = 1024

    with torch.no_grad():
        for name in pbar:
            images = []
            try:  
                # Image size: 1+num_attempts, 3, 1024, 1024
            
                img = Image.open(f"{name}.png", 'r')

                img_src = toTorch(crop(img, top=2, left=2 , height=size, width = size)).to(device)
                img_src = img_src.unsqueeze(0)
                img_tgt = toTorch(crop(img, top=2, left=size+4 , height=size, width =size)).to(device)
                img_tgt = img_tgt.unsqueeze(0)
                id_loss = criteria(img_src, img_tgt)[0]

                if args.conditioned and id_loss > 0.35:
                    continue

                img = toTorch(crop(img, top=2, left=size+4, height=size, width = size)).to(device)
                images.append(img)
    
            except OSError:
                continue

            value = pairwise_lpips(lpips_alex, images)
            if value == -1:
                continue
            values.append(value.detach().cpu()[0][0][0][0])

            for img in images:
                del img
            del images
            pbar.set_description(f"Processing {name}")

    mean_value = sum(values) / len(values)
    print(f"Final LPIPS : {mean_value}")
    if args.wandb:
        wandb.log({"lpips":mean_value})
        wandb.finish()

    


