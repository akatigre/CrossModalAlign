import pickle
import torch
import clip
from pathlib import Path
import json


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
imagenet_path = Path("../dataset/imagenet/")

with open(imagenet_path/"imagenet_class_index.json", 'r') as f:
    class_file = json.load(f)

with open("CrossModalRetrieval/semantics/imagenet.pkl", 'rb') as f:
    ANCHOR_DICT = pickle.load(f)
CANDIDATES = []
ANCHOR = []
#IMAGES = 
for class_name, embedding in ANCHOR_DICT.items():
    CANDIDATES.append(class_file[class_name])
    ANCHOR.append(embedding)


def retrieve(query):
    retrieval_emb = []
    txt_emb = []

    prototype_names = []
    for idx, description in enumerate(ANCHOR):
        
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

    with open(PATH / "ANCHOR.txt", 'w') as f:
        for line in prototype_names:
            f.write(line)

    retrieval_emb = torch.cat(retrieval_emb, dim=0)
    torch.save(retrieval_emb, PATH/"attr2img_embeddings.pt")

    txt_emb = torch.cat(txt_emb, dim=0)
    torch.save(txt_emb, PATH/"attr2txt_embeddings.pt")
