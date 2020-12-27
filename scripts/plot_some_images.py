from architectures.autoencoder import Generator as Autoencoder
from architectures.decoder import Generator as Decoder
from architectures.variational_autoencoder import Generator as Variational_Autoencoder
from dataset.TrackerDS_advanced import TrackerDS_advanced
from engine import *
from transforms import itk_transforms as itktransforms
from GAN_generators import Conditional_Decoder
from GAN_generators import Conditional_Adversarial_Autoencoder
from transforms import tensor_transforms as tensortransforms
from torchvision import transforms as torchtransforms
import torch
import numpy as np
import glob
from matplotlib import pyplot as plt
#import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter

#TRANSFORMS
resample = itktransforms.Resample(new_spacing=[.5, .5, 1.])
tonumpy  = itktransforms.ToNumpy(outputtype='float')
totensor = torchtransforms.ToTensor()
crop     = tensortransforms.CropToRatio(outputaspect=1.)
resize   = tensortransforms.Resize(size=(256,256), interp='BILINEAR')
rescale  = tensortransforms.Rescale(interval=(0,1))
transform = torchtransforms.Compose([resample, tonumpy, totensor, crop, resize, rescale])

ConstantRescale = tensortransforms.ConstantRescale(scaling = np.array([1/250,1/250,1/500,1.,1.,1.,1.]))

#DATASET AND DATALOADER
root = "/Users/cesaremagnetti/Documents/BEng_project/data/phantom_04Mar2020"

#root = "/home/cm19/BEng_project/data/data"
ds = TrackerDS_advanced(root, mode = "validate", transform=transform, target_transform=ConstantRescale)# ,reject_inside_radius=False, X_point=(-30,-0,-350), X_radius=30)
ds.remove_corrupted_indexes(ds.get_corrupted_indexes())

#CUDA SETUP
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
tensortype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

roots = [
    "/Users/cesaremagnetti/projects/PyCharm/UltrasoundSimulator/final_models/phantom/DECODER/last.tar",
    #"/Users/cesaremagnetti/Desktop/GAN_models/DECODER/resize_conv/last_first.tar",
    # "/Users/cesaremagnetti/Desktop/GAN_models/DECODER/resize_conv/epoch50.tar",
    # "/Users/cesaremagnetti/Desktop/GAN_models/DECODER/resize_conv/epoch100.tar",
    "/Users/cesaremagnetti/Desktop/GAN_models/DECODER/resize_conv/epoch150.tar",
    # "/Users/cesaremagnetti/projects/PyCharm/UltrasoundSimulator/models/AUTOENCODER/final_models/phantom/"
    # "AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/GAN_training/noise_size_100/resize_conv/epoch100.tar",
    # "/Users/cesaremagnetti/projects/PyCharm/UltrasoundSimulator/models/AUTOENCODER/final_models/phantom/"
    # "AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/GAN_training/noise_size_100/resize_conv/epoch150.tar",
    # "/Users/cesaremagnetti/projects/PyCharm/UltrasoundSimulator/models/AUTOENCODER/final_models/phantom/"
    # "AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/GAN_training/noise_size_100/resize_conv/epoch200.tar",
    #"/Users/cesaremagnetti/Desktop/GAN_models/DECODER/resize_conv/epoch200.tar",
    # "/Users/cesaremagnetti/Desktop/GAN_models/DECODER/resize_conv/epoch250.tar",
    # "/Users/cesaremagnetti/Desktop/GAN_models/DECODER/resize_conv/epoch300.tar",
]

names = ["decoder", "GAN decoder"]

#names.append("guess 2")
names = np.array(names)
#names = []
nets = ()

for idx, root in enumerate(roots):
    print(idx)
    checkpoint = torch.load(root, map_location='cpu')
    cfg = checkpoint['model_info']['model_configuration']
    # cfg['interpolation_scheme'] = 'nearest'
    for key in cfg.keys():
        print("{} --> {}".format(key, cfg[key]))
    if "GAN" in root and "DECODER" in root:
        net = Conditional_Decoder(cfg['z_size'],cfg['noise_size'],cfg['latent_image_size']).to(device)
    elif "GAN" in root and "AUTOENCODER" in root:
        net = Conditional_Adversarial_Autoencoder(cfg['z_size'], cfg['noise_size'], cfg['latent_image_size']).to(device)
    elif "DECODER" in root or "PRETRAINED" in root:
        net = Decoder(cfg).to(device)
    elif "VARIATIONAL_AUTOENCODER" in root:
        net = Variational_Autoencoder(cfg).to(device)
    elif "AUTOENCODER" in root:
        net = Autoencoder(cfg).to(device)


    # 1. filter out unnecessary keys
    pretrained_dict = {}
    flag = True
    state_dict = checkpoint['model_state_dict'].items() if "GAN"\
                 not in root else checkpoint['netG_state_dict'].items()
    for k, v in state_dict:
        if k not in net.state_dict():
            if flag:
                print("the following parameters will be discarded:\n")
                flag = False
            print("{}: {}".format(k, v))
        else:
            pretrained_dict[k] = v
    #net.print_info(verbose=True)
    #load state dict
    net.load_state_dict(pretrained_dict)

    #net.print_info(True)
    net.eval()
    #names.append(net.get_info('model_name'))
    nets+=(net,)

#determine indices of images to show at each epoch
np.random.seed()
#np.random.seed(5) #2
idxs = np.random.randint(len(ds), size=(10,))
print(idxs)
#idxs = np.array([212,212,212,212,212,83,83,83,83,83])
#idxs = np.array([54,44,95,310,82,215,294,228,87,313])

#idxs = np.arange(100,110)
imgs,outs = [],[]
for i, net in enumerate(nets):
    out = []
    for idx in idxs:
        img,tracker = ds[idx]
        img,tracker = img.unsqueeze(0).to(device).type(tensortype), torch.tensor(tracker).unsqueeze(0).to(device).type(tensortype)
        if i == 0:
            imgs.append(img.detach().cpu().numpy().squeeze())
        out.append(net.decode(tracker).detach().cpu().numpy().squeeze())
    outs.append(out)


imgs = np.array(imgs)
IM = np.hstack(np.array(imgs))
OUTS = []
for out in outs:
    OUTS.append(np.hstack(np.array(out)))

IMAGE = np.vstack(np.vstack(np.array([IM,] + OUTS)))

cmap = plt.cm.gray
norm = plt.Normalize(vmin=0, vmax=1)
#plt.imsave("/Users/cesaremagnetti/Documents/BEng_project/images_thesis/some_images.png",cmap(norm(IMAGE)))
fig = plt.figure()
plt.imshow(cmap(norm(IMAGE)))
plt.axis("off")
plt.show()