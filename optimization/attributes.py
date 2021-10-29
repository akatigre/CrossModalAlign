#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Creates image features prototype embeddings and clip text embeddings
"""
from PIL import Image
from pathlib import Path
import clip
import numpy as np
import torch
import glob
import torch.nn.functional as F

test_hard = [
    "A smiling young woman with short blonde hair", 
    "She has wavy hair, high cheekbones and oval face. She is wearing lipstick", 
    "This woman is young and has blond hair", 
    "This old woman has big lips, pale skin and gray hair.",
    "This man has very small eyes and black hair",
    "She has black eyebrows and blonde hair"
    ]
def get_sim(text_embeddings, images_embeddings):
    with torch.no_grad():
        text_embeddings = text_embeddings.float()
        logit_scale = 100 # model.logit_scale.exp()
        logits_per_image = (logit_scale * images_embeddings.float() @ text_embeddings.t()).squeeze()
        logits_per_image = logits_per_image.cpu().numpy()
    return logits_per_image

if __name__=="__main__":
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    PATH = Path("latents/prototypes")
    candidates = []

    with open(PATH / "candidates.txt", 'r') as f:
        candidates = []
        for i in f.readlines():
            if i != "\n":
                candidates.append(i)
    print(len(candidates))
    images_embeddings = torch.load(PATH / 'images_embeddings.pt').float() #[202599, 512] Clip embeddings of CelebA images
    
    list_images = glob.glob("celebA/images/*.jpg")
    
    celebA_path = Path("celebA/")
   
    images_embeddings = images_embeddings.to(device)
    n_samples = 15

    retrieval_emb = []
    txt_emb = []
    model, preprocess = clip.load('ViT-B/32', device)
    
    prototype_names = []
    for idx, description in enumerate(candidates):
        
        text_inputs = torch.cat([clip.tokenize(description)]).to(device)
        target_embedding = model.encode_text(text_inputs)
        txt_emb.append(target_embedding)
        logits_per_image = get_sim(target_embedding, images_embeddings)
        best_photo_idx = np.argsort(logits_per_image)[::-1]
        best_photos = [(logits_per_image[i],i) for i in best_photo_idx]
        embs = [images_embeddings[best_photo_idx[i]] for i in range(n_samples)]
        name = f"{description}"
        # for i in range(0, n_samples, 5):
        #     img = Image.open(list_images[best_photo_idx[i]])
        #     img.save(f"load_images/{name}_{i}.jpg")

        prototype_names.append(name)
        embs = torch.stack(embs).mean(dim=0, keepdim=True) # n_samples, 512
        retrieval_emb.append(embs)
    
    with open(PATH / "candidates.txt", 'w') as f:
        for line in prototype_names:
            f.write(line)

    retrieval_emb = torch.cat(retrieval_emb, dim=0)
    torch.save(retrieval_emb, PATH/"attr2img_embeddings.pt")

    txt_emb = torch.cat(txt_emb, dim=0)
    torch.save(txt_emb, PATH/"attr2txt_embeddings.pt")