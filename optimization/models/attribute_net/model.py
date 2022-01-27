
import os
import argparse
from torch import nn
from torch import optim
from torchvision import models, transforms

import torch 


class AttributeNet(nn.Module):
    def __init__(self):
        super(AttributeNet, self).__init__()
        self.model = models.resnet18(pretrained=False, num_classes=2)
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))
        ])
        self.loss = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", default=2)
    parser.add_argument("--image_size", default=224)
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--lr", default=1e-2)
    parser.add_argument("--pretrained", action="store_true", help="Loads ImageNet pretrained weight for resnet18")
    parser.add_argument("--data_path", default="celebA/")
    parser.add_argument("--ckpt_path", default="celebA/manip_quality/ckpt/")
    args = parser.parse_args()

    args.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))
        ])

    choices=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',\
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',\
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',\
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',\
        'Male', 'Mouth_Slightly_Open','Wearing_Necktie', 'Young']

    for attr in choices:
        args.attribute = attr
        model = AttributeNet()

        print(attr)
        model_path = os.path.join("../../../ckpts", f"{attr}.pt")
        try:
            model.load_from_checkpoint(model_path)
        except KeyError:
            print("Already finished")
            continue
       
        torch.save(model.state_dict(),model_path)
