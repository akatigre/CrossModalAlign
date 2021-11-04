# Creates averaged clip embeddings for each class of ImageNet

import os
import pickle
import torch
import glob
import clip
from PIL import Image
from torchvision.datasets.folder import ImageFolder
from torch.nn import functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
data_dir = "../dataset/imagenet/"
with open("./semantics/imagenet.pkl", 'rb') as f:
    dt = pickle.load(f)

none_class = []
for class_name, clip_embedding in dt.items():
    if clip_embedding is None:
        none_class.append(class_name)

traindir = os.path.join(data_dir, 'train')
classes = sorted(ImageFolder(traindir).classes)
class_folders = sorted(glob.glob(traindir+"/*/"))


n = 0
for idx, class_folder in enumerate(class_folders):
    
    class_name = str(class_folder).split('/')[-2]
    if class_name not in none_class:
        print("pass")
        continue
    n+=1
    print(f"{class_folder} {n}/327")
    imgs = glob.glob(class_folder+"*.JPEG")[:100]
    # feature = torch.zeros((1, 512)).to(device)
    img_list = []
    for img in imgs:
        image = Image.open(img)
        image_input = preprocess(image).unsqueeze(0).to(device)
        img_list.append(image_input)
    image_input = torch.cat(img_list)
    with torch.no_grad():
        feature = model.encode_image(image_input).mean(dim=0, keepdim=True).detach().cpu().float()
    dt[classes[idx]] = F.normalize(feature, p=2, dim=1)
    with open("semantics/imagenet" + '.pkl', 'wb') as f:
        pickle.dump(dt, f)
        


# def batchify(thing, n):            
#     toexit = False
#     it = iter(thing)
#     while not toexit:
#         batch = []
#         for _ in range(n):
#             try:
#                 batch.append(next(it))
#             except StopIteration:
#                 toexit = True
#         if not batch:
#             break
#         yield batch
