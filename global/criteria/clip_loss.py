import torch
import clip

class CLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity

    def encode_text(self, text):
        tokenized = torch.cat([clip.tokenize(text)]).cuda()
        text_features = self.model.encode_text(tokenized.long())
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()

    def encode_image(self, image):
        image = self.avg_pool(self.upsample(image))
        image_features = self.model.encode_image(image)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        return image_features.float()