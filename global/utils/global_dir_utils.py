import clip
import copy
import numpy as np
import pickle
import torch
from utils.stylegan_models import encoder, decoder

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
    # eps = 1.0
    for idx in idxs:
        idx = idx.detach().cpu()
        # ds_imp[idx] = tmp[idx] + np.random.normal(0.0, eps, (1))
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

def MSCode2(dlatent_tmp, boundary_tmp, boundary_tmp2, alpha, beta, device):
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
    
    l2=np.array(beta)
    l2=l2.reshape([step if axis == 1 else 1 for axis in range(dlatent_tmp2[0].ndim)])
    
    tmp=np.arange(len(boundary_tmp))
    for i in tmp:
        dlatent_tmp2[i]+=l*boundary_tmp[i] + l2*boundary_tmp2[i]
    
    codes=[]
    for i in range(len(dlatent_tmp2)):
        tmp=list(dlatent_tmp[i].shape)
        tmp.insert(1,step)
        code = torch.Tensor(dlatent_tmp2[i].reshape(tmp))
        codes.append(code.cuda())
    return codes

def zeroshot_classifier(classnames, model):
    """
    model: CLIP 
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in imagenet_templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def create_dt(target, model, neutral=""):
    text_features = zeroshot_classifier([target, neutral], model).T
    dt = text_features[0]-text_features[1]
    dt = dt / dt.norm()
    return dt.unsqueeze(0).float()

def create_image_S(generator, latent):
    with torch.no_grad():
        style_space, style_names, noise_constants = encoder(generator, latent)
        img_orig = decoder(generator, style_space, latent, noise_constants)
    return img_orig, style_space, style_names, noise_constants

def manipulate_image(style_space, style_names, noise_constants, generator, latent, args, alpha=5, t=None, s_dict=None, device="cuda:0"):
    boundary_tmp2, _, _, _ = GetBoundary(s_dict, t.squeeze(axis=0), args, style_space, style_names) # Move each channel by dStyle
    dlatents_loaded = [s.cpu().detach().numpy() for s in style_space]
    manip_codes= MSCode(dlatents_loaded, boundary_tmp2, [alpha], device)
    img_gen = decoder(generator, manip_codes, latent, noise_constants)
    return img_gen, manip_codes, style_space

def manipulate_image2(style_space, style_names, noise_constants, generator, latent, args, alpha=5, beta=5, t=None, t2= None, s_dict=None, device="cuda:0"):
    boundary_tmp2, _, _, _ = GetBoundary(s_dict, t.squeeze(axis=0), args, style_space, style_names) # Move each channel by dStyle
    boundary_tmp22, _, _, _ = GetBoundary(s_dict, t2.squeeze(axis=0), args, style_space, style_names) # Move each channel by dStyle
    dlatents_loaded = [s.cpu().detach().numpy() for s in style_space]
    manip_codes= MSCode2(dlatents_loaded, boundary_tmp2, boundary_tmp22, [alpha], [beta], device)
    img_gen = decoder(generator, manip_codes, latent, noise_constants)
    return img_gen, manip_codes, style_space
