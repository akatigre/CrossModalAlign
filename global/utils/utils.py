import pickle
import numpy as np
import clip
import torch
import copy 
import cv2
import numpy as np
from functools import partial
from torch.nn import functional as F
from sklearn.decomposition import PCA
import torchvision.transforms as transforms


l2norm = partial(F.normalize, p=2, dim=1)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def conv_warper(layer, input, style, noise):
    # the conv should change
    conv = layer.conv
    batch, in_channel, height, width = input.shape
    style = style.view(batch, 1, in_channel, 1, 1)
    weight = conv.scale * conv.weight * style

    if conv.demodulate:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

    weight = weight.view(
        batch * conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
    )

    if conv.upsample: # up==2
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, conv.out_channel, conv.kernel_size, conv.kernel_size
        )
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        out = conv.blur(out)

    elif conv.downsample:
        input = conv.blur(input)
        _, _, height, width = input.shape
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    else:
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=conv.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
    out = layer.noise(out, noise=noise)
    out = layer.activate(out)
    
    return out

def decoder(G, style_space, latent, noise):
    """
    Returns array of generated image from manipulated style space
    """

    out = G.input(latent)

    out = conv_warper(G.conv1, out, style_space[0], noise[0])
    skip = G.to_rgb1(out, latent[:, 0])

    i = 2; j = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        out = conv_warper(conv1, out, style_space[i], noise=noise1)
        out = conv_warper(conv2, out, style_space[i+1], noise=noise2)
        skip = to_rgb(out,  latent[:, j + 2], skip)

        i += 3; j += 2

    image = skip

    return image

def encoder(G, latent): 
    noise_constants = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]
    style_space = []
    style_names = []
    # rgb_style_space = []
    style_space.append(G.conv1.conv.modulation(latent[:, 0]))
    res=4
    style_names.append(f"b{res}/conv1")
    style_space.append(G.to_rgbs[0].conv.modulation(latent[:, 0]))
    style_names.append(f"b{res}/torgb")
    i = 1; j=3
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise_constants[1::2], noise_constants[2::2], G.to_rgbs
    ):
        res=2**j
        style_space.append(conv1.conv.modulation(latent[:, i]))
        style_names.append(f"b{res}/conv1")
        style_space.append(conv2.conv.modulation(latent[:, i+1]))
        style_names.append(f"b{res}/conv2")
        style_space.append(to_rgb.conv.modulation(latent[:, i + 2]))
        style_names.append(f"b{res}/torgb")
        i += 2; j += 1
        
    return style_space, style_names, noise_constants


def projection(basis, target, multiple=False):
    B = basis.detach().cpu()
    X = target.detach().cpu()
    
    if multiple:
        inv_B = torch.solve(B, torch.matmul(B, B.T)).solution
        P = torch.matmul(B.T, inv_B)
        return l2norm(torch.matmul(X, P)).cuda()
    else:
        B = B.squeeze(0)
        X = X.squeeze(0)
        return l2norm((X.dot(B.T)/B.dot(B) * B).unsqueeze(0)).cuda()


def ffhq_style_semantic(channels):
    configs_ffhq = {
    'black hair' :      [(12, 479)],
    'blond hair':      [(12, 479), (12, 266)],
    'grey hair' :      [(11, 286)],
    'wavy hair'  :      [(6, 500), (8, 128), (5, 92), (6, 394), (6, 323)],
    'bangs'      :      [(3, 259), (6, 285), (5, 414), (6, 128), (9, 295), (6, 322), (6, 487), (6, 504)],
    'receding hairline':[(5, 414), (6, 322), (6, 497), (6, 504)],
    'smiling'    :      [(6, 501)],
    'lipstick'   :      [(15, 45)],
    'sideburns'  :      [(12, 237)],
    'goatee'     :      [(9, 421)],
    'earrings'   :      [(8, 81)],
    'glasses'    :      [(3, 288), (2, 175), (3, 120), (2, 97)],
    'wear suit'  :      [(9, 441), (8, 292), (11, 358), (6, 223)],
    'gender'     :      [(9, 6)]
    }
    style_channels = []
    for res, num_channels in channels.items():
        if res==4:
            style_channels.append(num_channels)
        else:
            style_channels.extend([num_channels]*2)
    
    mapped = {}
    for k, v in configs_ffhq.items():
        new_v = [sum(style_channels[:layer]) + ch for layer, ch in v]
        mapped[k] = new_v
    return mapped

imagenet_templates = [
    'a bad photo of a {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

def GetTemplate(target, model):
    """
    model: CLIP 
    """
    with torch.no_grad():
        texts = [template.format(target) for template in imagenet_templates] #format with class
        texts = clip.tokenize(texts).cuda() #tokenize
        class_embeddings = model.encode_text(texts) #embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
    return class_embedding


def uniform_loss(x, t=2):
    x = torch.Tensor(x).cuda()
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def logitexp(logp):
    # Convert outputs of logsigmoid to logits (see https://github.com/pytorch/pytorch/issues/4007)
    pos = torch.clamp(logp, min=-0.69314718056)
    neg = torch.clamp(logp, max=-0.69314718056)
    neg_val = neg - torch.log(1 - torch.exp(neg))
    pos_val = -torch.log(torch.clamp(torch.expm1(-pos), min=1e-20))
    return pos_val + neg_val

def disentangle_fs3(fs3):
    pca = PCA(n_components=5)
    mean = fs3.mean()
    fs3_tmp = (fs3 - mean)
    pca.fit(fs3_tmp)
    comp = np.matmul(np.matmul(fs3, pca.components_.T), pca.components_) # projection
    return fs3_tmp - comp
    

def GetBoundary(fs3, dt, args, style_space, style_names):
    """
    fs3: collection of predefined style directions for each channel (6048, 512)
    """
    tmp = np.dot(fs3, dt)
    if args.topk == 0: 
        ds_imp = copy.copy(tmp)
        select = np.abs(tmp)< args.beta
        num_c = np.sum(~select)
        ds_imp[select] = 0
        tmp = np.abs(ds_imp).max()
        ds_imp /=tmp

        boundary_tmp2, dlatents = SplitS(ds_imp, style_names, style_space, args.nsml)
        print('num of channels being manipulated:',num_c)
        return boundary_tmp2, num_c, dlatents, []

    num_c = args.topk
    _, idxs = torch.topk(torch.Tensor(np.abs(tmp)), num_c)
    ds_imp = np.zeros_like(tmp)
    for idx in idxs:
        idx = idx.detach().cpu()
        ds_imp[idx] = tmp[idx]
    tmp = np.abs(ds_imp).max()
    ds_imp/=tmp
    boundary_tmp2, dlatents=SplitS(ds_imp, style_names, style_space, args.nsml)
    print('num of channels being manipulated:',num_c)
    return boundary_tmp2, num_c, dlatents, idxs[:5]
        
def SplitS(ds_p, style_names, style_space, nsml=False):
    """
    Split array of 6048(toRGB ignored) channels into corresponding channel size (into 9088)
    """
    all_ds=[]
    start=0
    dataset_path = "./npy/ffhq/" if not nsml else "./global/npy/ffhq/"
    tmp=dataset_path+'S'
    with open(tmp, "rb") as fp:
        _, dlatents=pickle.load( fp)
    tmp=dataset_path+'S_mean_std'
    with open(tmp, "rb") as fp:
        m, std=pickle.load( fp)

    for i, name in enumerate(style_names):
        if "torgb" not in name:
            tmp=style_space[i].shape[1]
            end=start+tmp
            tmp=ds_p[start:end] * std[i]
            all_ds.append(tmp)
            start=end
        else:
            tmp = np.zeros(len(dlatents[i][0]))
            all_ds.append(tmp)
    return all_ds, dlatents

def MSCode(dlatent_tmp, boundary_tmp, alpha, device):
    """
    dlatent_tmp: W mapped into style space S
    boundary_tmp: Manipulation vector
    Returns:
        manipulated Style Space
    """
    step=len(alpha)
    dlatent_tmp1=[tmp.reshape((1,-1)) for tmp in dlatent_tmp]
    dlatent_tmp2=[np.tile(tmp[:,None],(1,step,1)) for tmp in dlatent_tmp1]

    l=np.array(alpha)
    l=l.reshape([step if axis == 1 else 1 for axis in range(dlatent_tmp2[0].ndim)])
    
    tmp=np.arange(len(boundary_tmp))
    for i in tmp:
        dlatent_tmp2[i]+=l*boundary_tmp[i]
    
    codes=[]
    for i in range(len(dlatent_tmp2)):
        tmp=list(dlatent_tmp[i].shape)
        tmp.insert(1,step)
        code = torch.Tensor(dlatent_tmp2[i].reshape(tmp))
        codes.append(code.cuda())
    return codes

def Text2Prototype(target):
    print(target)
    prototype_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    prototype_list = [idx.lower() for idx in prototype_list]
    target = target.lower()
    target = target.replace(' ', '_')
    print(target)
    if target in prototype_list:
        return target
    else:
        return None

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

