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
from itertools import product
import glob


test_easy = [
    "Light Skin",
    "Dark Skin",
    "Arched Eyebrows",
    "Straight Eyebrows",
    "Bags under eyes",
    "Bangs",
    "Big lips",
    "Small lips",
    "Pointy Nose",
    "Big Nose",
    "Small Nose",
    "Black Hair",
    "Blond Hair",
    "Brown Hair",
    "Red Hair",
    "Grey Hair",
    "Bushy Eyebrows",
    "Double Chin",
    "Goatee",
    "Heavy makeup",
    "No makeup",
    "High cheekbones",
    "Male",
    "Female",
    "Mouth open",
    "Mouth closed",
    "Mustache",
    "beard",
    "No facial hair",
    "Oval face",
    "Wide eyes",
    "Sleepy eyes",
    "Rosy cheeks",
    "Smiling",
    "Frowning",
    "Straight hair",
    "Wavy hair",
    "Short hair",
    "Long hair",
    "Wearing lipstick",
    "Young",
    "Old"
]
TediGAN = [
    "A smiling young woman with short blonde hair", 
    "He is young and wears beard", 
    "A young woman with long black hair",
    "This man has bags under eyes and big nose. He has no beard",
    "She has wavy hair, high cheekbones and oval face. She is wearing lipstick", 
    "This woman is young and has blond hair", 
    "She has oval face and long black hair. She is wearing earrings", 
    "He has no beard", 
    "This old woman has big lips, pale skin and gray hair.", 
    "This woman wears earrings. She has oval face and high bones. She is smiling.",
    "She wears eyeglasses"
        ]
celebA_text = [
    '5_o_Clock_Shadow', 
    'Arched_Eyebrows', 
    'Attractive', 
    'Bags_Under_Eyes',
    'Bald', 
    'Bangs', 
    'Big_Lips', 
    'Big_Nose', 
    'Black_Hair', 
    'Blond_Hair',
    'Blurry', 
    'Brown_Hair', 
    'Bushy_Eyebrows', 
    'Chubby', 
    'Double_Chin',
    'Eyeglasses', 
    'Goatee', 
    'Gray_Hair', 
    'Heavy_Makeup', 
    'High_Cheekbones',
    'Male', 
    'Mouth_Slightly_Open', 
    'Mustache', 
    'Narrow_Eyes', 
    'Beard',
    'Oval_Face', 
    'Pale_Skin', 
    'Pointy_Nose', 
    'Receding_Hairline',
    'Rosy_Cheeks', 
    'Sideburns', 
    'Smiling', 
    'Straight_Hair', 
    'Wavy_Hair',
    'Wearing_Earrings', 
    'Wearing_Hat', 
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young'
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

    images_embeddings = torch.load(PATH / 'images_embeddings.pt') #[202599, 512] Clip embeddings of CelebA images
    list_images = glob.glob("celebA/images/*.jpg")
    
    celebA_path = Path("celebA/")
   
    images_embeddings = images_embeddings.to(device)
    n_samples = 30

    retrieval_emb = []
    txt_emb = []
    model, preprocess = clip.load('ViT-B/32', device)
    
    candidates = []   
    for idx, description in enumerate(celebA_text):
        for dir in [1, -1]:
            text_inputs = torch.cat([clip.tokenize(description)]).to(device)
            target_embedding = model.encode_text(text_inputs)
            txt_emb.append(target_embedding)
            logits_per_image = get_sim(dir * target_embedding, images_embeddings)
            best_photo_idx = np.argsort(logits_per_image)[::-1]
            best_photos = [(logits_per_image[i],i) for i in best_photo_idx]
            embs = [images_embeddings[best_photo_idx[i]] for i in range(n_samples)]

            tag = "pos" if dir==1 else "neg"
            name = f"{description}_{tag}"
            for i in range(0, 30, 5):
                img = Image.open(list_images[best_photo_idx[i]])
                img.save(f"load_images/{name}_{i}.jpg")

            candidates.append(name)
            embs = torch.stack(embs).mean(dim=0, keepdim=True) # n_samples, 512
            retrieval_emb.append(embs)
    
    with open(PATH / "candidates.txt", 'w') as f:
        for line in candidates:
            f.write(line)
            f.write("\n")
            
    retrieval_emb = torch.cat(retrieval_emb, dim=0)
    torch.save(retrieval_emb, PATH/"attr2img_embeddings.pt")

    txt_emb = torch.cat(txt_emb, dim=0)
    torch.save(txt_emb, PATH/"attr2txt_embeddings.pt")