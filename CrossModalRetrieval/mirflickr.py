
from pathlib import Path
from collections import defaultdict
mir_path = Path("../dataset/mirflickr")
labels = mir_path / "annotations"
tags = mir_path / "tags"
imgs = mir_path / "imgs"
anno = []
for file in list(labels.glob("*")):
    if "_r1" not in str(file) and str(file)!="README.txt":
        anno.append(str(file).split("/")[-1][:-4])


# Collection of tags
tag_img = defaultdict(int)
tag_files = list(tags.glob("*"))
print(len(tag_files))
for file in tag_files:
    with open(file, 'r') as f:
        for i in f.readlines():
            i = i.strip("\n")
            tag_img[i] += 1

def filter_words(word):
    numb = any(not char.isdigit() for char in word)
    stopwords = ["flickr"]
    stop = any(True for stopword in stopwords if stopword in word)
    length = len(word)<4
    return any([numb, stop, length])

valid_tags = [k for k, v in tag_img.items() if 20<v and filter_words(k)]
print(valid_tags)
