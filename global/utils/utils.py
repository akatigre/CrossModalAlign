import copy
import pickle
import numpy as np
import clip
import torch
import numpy as np
from functools import partial
from torch.nn import functional as F
from sklearn.decomposition import PCA


l2norm = partial(F.normalize, p=2, dim=1)

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
        inv = torch.matmul(B, B.T)
        X = torch.matmul(B, X.T)
        P, _ = torch.solve(inv, X)
        proj = torch.matmul(P.T, B)
        return l2norm(proj).cuda()
    else:
        B = B.squeeze(0)
        X = X.squeeze(0)
        return l2norm((X.dot(B.T)/B.dot(B) * B).unsqueeze(0)).cuda()

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
    if args.beta != 0:
        threshold = args.beta
        ds_imp=copy.copy(tmp)
        select = np.abs(tmp)<threshold
        num_c = np.sum(~select)
        ds_imp[select] = 0
    else:
        num_c = args.topk
        _, idxs = torch.topk(torch.Tensor(np.abs(tmp)), num_c)
        ds_imp = np.zeros_like(tmp)
        for idx in idxs:
            idx = idx.detach().cpu()
            ds_imp[idx] = tmp[idx]
    tmp = np.abs(ds_imp).max()
    ds_imp/=tmp
    boundary_tmp2, dlatents=SplitS(ds_imp, style_names, style_space)
    print('num of channels being manipulated:',num_c)
    return boundary_tmp2, num_c, dlatents
        
def SplitS(ds_p, style_names, style_space):
    """
    Split array of 6048(toRGB ignored) channels into corresponding channel size (into 9088)
    """
    all_ds=[]
    start=0
    tmp="./npy/ffhq/"+'S'
    with open(tmp, "rb") as fp:
        _, dlatents=pickle.load( fp)
    tmp="./npy/ffhq/"+'S_mean_std'
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