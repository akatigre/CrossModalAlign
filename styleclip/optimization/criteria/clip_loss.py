import torch
import clip

class CLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda:0")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

    def forward(self, image, text):
        image_features = image
        text_features = text
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        similarity = 1 - logits_per_text[0] / 100
        return similarity

    def encode_text(self, text):
        tokenized = torch.cat([clip.tokenize(text)]).cuda()
        text_features = self.model.encode_text(tokenized.long())
        text_feature = text_features / text_features.norm(dim=-1, keepdim=True).float()
        return text_feature

    def encode_image(self, image):
        image = self.avg_pool(self.upsample(image))
        image_features = self.model.encode_image(image)
        image_feature = image_features/image_features.norm(dim=-1, keepdim=True).float()
        return image_feature