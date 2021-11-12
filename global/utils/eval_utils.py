import cv2
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
)
import torch
import torchvision.transforms as transforms
from pathlib import Path
import glob
import clip
from PIL import Image
from criteria.clip_loss import CLIPLoss
import argparse
### Evaluate diversity of segments ###
def Text2Segment(target):
    # return pairs of segments 
    # segment-wise (~19 Label)
    # 0: 배경, 1 : face, 2 : 오른쪽 눈썹, 3: 왼쪽 눈썹, 4: 오른쪽 눈, 5: 왼쪽 눈 
    # 7 : 오른쪽 귀, 8: 왼쪽 귀, 10: 코, 12: 윗 입술, 13: 아래 입술, 14: 목, 16: 옷, 17: 머리카락
    # 6 : 안경, 9 : 귀걸이, 11 : 치아(mouth),  15 : 목걸이, 18 : 모자 
    target = target.lower()
    dict = {"arched eyebrows":[2,3], "bushy eyebrows":[2, 3], "lipstick":[12, 13], "Eyeglasses":[6], 
            "bangs":[17], "black hair":[17], "blond hair":[17], "straight hair":[17], "Earrings":[9], "sideburns":[17], "Goatee":[17], "Receding hairline":[17], "Grey hair":[17], "Brown hair":[17],
            "wavy hair":[17], "wear suit":[16], "wear lipstick":[12, 13], "double chin":[1], "hat":[18], "Big nose":[10], "big lips":[12, 13], "High cheekbones":[1]}
    if target not in dict.keys():
        return []

    return dict[target]

def maskImage(img, Segment_net, device, segments, stride):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    resize = transforms.Resize((512, 512))

    # save_image(img, "before_mask.png", normalize=True, range=(-1, 1))
    img = resize(img)
    parses = Segment_net(img)[0]
    parses = parses.squeeze(0).cpu().numpy().argmax(0)
    # save_image(img, "mask.png", normalize=True, range=(-1, 1))

    vis_im = img.cpu()
    vis_parsing_anno = parses.copy().astype(np.uint8) # [512, 512] - size of the image 
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    num_of_class = np.max(vis_parsing_anno)
    check = False

    for pi in range(0, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        if pi not in segments: 
            check = True
            vis_im[:, :, index[0], index[1]] = 0

    # save_image(vis_im, "after_mask.png", normalize=True, range=(-1, 1))

    if not check: 
        return None

    vis_im = vis_im.to(device)
    return vis_im  

def get_sim(text_embeddings, images_embeddings):
    with torch.no_grad():
        text_embeddings = text_embeddings.float()
        logit_scale = 100 # model.logit_scale.exp()
        logits_per_image = (logit_scale * images_embeddings.float() @ text_embeddings.t()).squeeze()
        logits_per_image = logits_per_image.cpu().numpy()
    return logits_per_image

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    args = parser.parse_args()
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    PATH = Path("../npy/")

    list_images = glob.glob("../../../dataset/celebA/images/*.jpg")
    model, preprocess = clip.load("ViT-B/32", device=device)
    # images_embeddings = torch.load(PATH/"celeba_clip.pt").to(device)
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    loader = batch(list_images, n=10)

    embs = []
    for b in loader:
        imgs = []
        for img in b:
            img = preprocess(Image.open(img))
            imgs.append(img)
        imgs = torch.stack(imgs).to(device)
        print(imgs.shape)
        imgs = imgs.squeeze(0) if imgs.dim==5 else imgs
        with torch.no_grad():
            emb = model.encode_image(imgs)
        embs.append(emb.detach().cpu())
    images_embeddings = torch.stack(embs, dim=-1).to(device)
    torch.save(images_embeddings, "clip_celeba.pt")
    retrieval_emb = []
    txt_emb = []
    retrieval_texts = ["red hair"]
    candidates = []   
    for idx, description in enumerate(retrieval_texts):
        text_inputs = torch.cat([clip.tokenize(description)]).to(device)
        target_embedding = model.encode_text(text_inputs)
        txt_emb.append(target_embedding)
        logits_per_image = get_sim(target_embedding, images_embeddings)
        best_photo_idx = np.argsort(logits_per_image)[::-1]
        best_photos = [(logits_per_image[i],i) for i in best_photo_idx]
        embs = [images_embeddings[best_photo_idx[i]] for i in range(n_samples)]
        for i in range(0, 30, 10):
            img = Image.open(list_images[best_photo_idx[i]])
            img.save(f"{description}_{i}.jpg")

        embs = torch.stack(embs)
        torch.save(embs, f"{description}.pt")