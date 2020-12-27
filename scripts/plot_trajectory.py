from architectures.autoencoder import Generator as Autoencoder
from architectures.decoder import Generator as Decoder
from architectures.variational_autoencoder import Generator as Variational_Autoencoder
from dataset.TrackerDS_advanced import TrackerDS_advanced
from engine import *
from transforms import itk_transforms as itktransforms
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

#CUDA SETUP
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
tensortype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


#DATASET AND DATALOADER#########################################
root = "/Users/cesaremagnetti/Documents/BEng_project/data/phantom_04Mar2020"
#root = "/home/cm19/BEng_project/data/data"
ds = TrackerDS_advanced(root, mode = "infer", transform=transform, target_transform=ConstantRescale)
ds.remove_corrupted_indexes(ds.get_corrupted_indexes())
#############################################################

#load networks###############################################
roots = [
    "/users/cesaremagnetti/final_models/phantom/DECODER/last.tar",
    "/users/cesaremagnetti/final_models/phantom/PRETRAINED/last.tar",
    "/users/cesaremagnetti/final_models/phantom/VARIATIONAL_AUTOENCODER/last.tar",
    "/users/cesaremagnetti/final_models/phantom/AUTOENCODER/last.tar"
]
nets = ()
for idx, root in enumerate(roots):
    print(idx)
    checkpoint = torch.load(root, map_location='cpu')
    cfg = checkpoint['model_info']['model_configuration']
    # cfg['interpolation_scheme'] = 'nearest'
    for key in cfg.keys():
        print("{} --> {}".format(key, cfg[key]))
    if "DECODER" in cfg['net_name']:
        net = Decoder(cfg).to(device)
    elif "VARIATIONAL_AUTOENCODER" in cfg['net_name']:
        net = Variational_Autoencoder(cfg).to(device)
    elif "AUTOENCODER" in cfg['net_name']:
        net = Autoencoder(cfg).to(device)
    else:
        net = Decoder(cfg).to(device)
        print("WARNING: inconsistent name for root directory:\n{}\n, decoder architecture was instanciated by default,"\
              " rename the root directory accordingly if needed:\n i.e. ~/ANY PATH/ARCHITECTURE TYPE (egs: DECODER or"\
              " AUTOENCODER)/ANY PATH/model.tar")

    # 1. filter out unnecessary keys
    pretrained_dict = {}
    flag = True
    for k, v in checkpoint['model_state_dict'].items():
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
#############################################################


#get full trajectory#########################################
x, y, z = [], [], []
for i in range(len(ds)):
    _, tr = ds[i]
    x.append(tr[0])
    y.append(tr[1])
    z.append(tr[2])
    if i%100 ==0:
        print("{}%".format(i/len(ds)*100))
#############################################################

#get a trajectory of 10 samples##############################
N = 10
idx = np.random.randint(len(ds)-N)
idx = 852
imgs,outs = [],[]
for j in range(len(nets)):
    outs.append([])
for i in range(N):
    im,tr = ds[idx+i]
    imgs.append(im.detach().cpu().numpy().squeeze())
    tr = torch.tensor(tr).unsqueeze(0).to(device).type(tensortype)
    for j in range(len(nets)):
        outs[j].append(nets[j].decode(tr).detach().cpu().numpy().squeeze())

IM = np.hstack(imgs)
for j in range(len(outs)):
    outs[j] = np.hstack(outs[j])

IMAGE = np.vstack([IM,]+outs)
cmap = plt.cm.gray
norm = plt.Normalize(vmin=0, vmax=1)
fig = plt.figure(figsize=(15,5))
#plt.set_cmap("Set3")
grid = plt.GridSpec(5, 15, hspace=0., wspace=0.)

points = fig.add_subplot(grid[:,:5])
images = fig.add_subplot(grid[:,5:])

points.scatter(y,z, c = 'k', alpha = 0.5)
trajectory = points.scatter(y[idx:idx+N],z[idx:idx+N],c = np.arange(N),cmap=plt.cm.get_cmap('Set3', N))
fig.colorbar(trajectory, ax = points)
points.set_xlabel("Y (mm)")
points.set_ylabel("Z (mm)")
points.set_title("highlighted trajectory")
# norm = matplotlib.colors.Normalize(vmin=0, vmax=N-1)
# fig.colorbar(SM(norm=norm, cmap=plt.get_cmap("Set3")), ax = points)

images.imshow(cmap(norm(IMAGE)))
names = ["original", "decoder", "pretrained\ndecoder", "autoencoder", "variational\nautoencoder"]
images.set_xticks([128+idx*256 for idx in range(N)])
images.set_xticklabels([idx+1 for idx in range(N)])
images.set_yticks([128+idx*256 for idx in range(len(nets)+1)])
images.set_yticklabels([names[idx] for idx in range(len(names))])

print(idx)
plt.tight_layout()
plt.show()