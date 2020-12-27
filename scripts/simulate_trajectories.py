# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""
script to generate a simulated trajectory from a ground truth trajectory
"""
# Author:
# Cesare Magnetti <cesare.magnetti98@gmail.com> <cesare.magnetti@kcl.ac.uk>
#    King's College London, UK

from architectures.decoder import Generator as Decoder
from architectures.autoencoder import Generator as Autoencoder
from architectures.variational_autoencoder import Generator as Variational_Autoencoder
import tensor_transforms  as tensortransforms
import itk_transforms as itktransforms
from torchvision import transforms as torchtransforms
from TrackerDS_advanced import TrackerDS_advanced
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#TRANSFORMS
resample = itktransforms.Resample(new_spacing=[.5, .5, 1.])
tonumpy  = itktransforms.ToNumpy(outputtype='float')
totensor = torchtransforms.ToTensor()
crop     = tensortransforms.CropToRatio(outputaspect=1.)
resize   = tensortransforms.Resize(size=(256,256), interp='BILINEAR')
rescale  = tensortransforms.Rescale(interval=(0,1))
transform = torchtransforms.Compose([resample, tonumpy, totensor, crop, resize, rescale])
ConstantRescale = tensortransforms.ConstantRescale(scaling = np.array([1/250,1/250,1/500,1.,1.,1.,1.]))

#DATASET AND DATALOADER##################################
root = "/Users/cesaremagnetti/Documents/BEng_project/data/phantom_04Mar2020"
ds = TrackerDS_advanced(root, "infer", transform = transform, target_transform=ConstantRescale)
ds.remove_corrupted_indexes(ds.get_corrupted_indexes())
#########################################################

#get a part of the original trajectory ######################
images, trackers=[],[]
idx = np.random.randint(len(ds))
idx = 1280
N = 10
skip = 1
color = np.arange(idx,idx+skip*(N),skip)
for i in range(idx,idx+skip*(N),skip):
    img, tr = ds[i]
    images.append(img.detach().cpu().numpy().squeeze())
    trackers.append(tr)
trackers=np.array(trackers)
images = np.array(images)
##############################################################

#simulate trajectory linearly#################################
TRACKERS = []
diff = (trackers[-1]-trackers[0])/(N-1)

for i in range(N):
    #add i to the z component (scaled accordingly!) for example
    new_tracker = trackers[0] + i*diff
    TRACKERS.append(new_tracker)

TRACKERS = np.array(TRACKERS)
assert TRACKERS.shape == trackers.shape #shoeld have same number of samples and features
##############################################################

#simulate trajectory quadratically#################################
TRACKERS = []
diff = (trackers[-1]-trackers[0])/(N-1)

for i in range(N):
    #add i to the z component (scaled accordingly!) for example
    new_tracker = trackers[0] + i*diff
    TRACKERS.append(new_tracker)

TRACKERS = np.array(TRACKERS)
assert TRACKERS.shape == trackers.shape #shoeld have same number of samples and features
##############################################################

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
fig = plt.figure(1)
ax = fig.add_subplot(1,2,1)
ax.scatter(y,z, c = 'k', alpha = 0.5)
ax.scatter(trackers[:,1],trackers[:,2],c = color)
plt.xlabel("Y (mm)")
plt.ylabel("Z (mm)")
ax.set_title("highlighted trajectory")

ax = fig.add_subplot(1,2,2)
ax.scatter(trackers[:,1],trackers[:,2], c = 'k', label = "original trajectory")
ax.scatter(TRACKERS[:,1],TRACKERS[:,2], c = 'r', label = "linearly interpolated trajectory")
plt.xlabel("Y (mm)")
plt.ylabel("Z (mm)")
ax.set_title("simulated trajectory (linear interpolation)")
plt.legend()
plt.show()

#LOAD MODELS#######################################################

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
tensor_type = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

root = "/Users/cesaremagnetti/Documents/BEng_project_repo/models/DECODER/DeeperDec/noDropout/MSELoss/" \
       "new_models/DECODER__LINEAR_7_512_1024_2048__CONV_32x6_1/resize_conv/last.tar"

checkpoint = torch.load(root, map_location='cpu')
cfg = checkpoint['model_info']['model_configuration']
for key in cfg.keys():
    print("{} --> {}".format(key, cfg[key]))
if "DECODER" in cfg['net_name']:
    net = Decoder(cfg).to(device)
elif "VARIATIONAL_AUTOENCODER" in cfg['net_name']:
    net = Variational_Autoencoder(cfg).to(device)
elif "AUTOENCODER" in cfg['net_name']:
    net = Autoencoder(cfg).to(device)

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

#load state dict
net.load_state_dict(pretrained_dict)

#net.print_info(True)
net.eval()

#GENERATE THE FINAL IMAGE: 1st row: real images, 2nd row: simulated images, 3rd row: linearly interpolated images through the simulation

IMAGES = np.hstack(images) #1st row
outs1,outs2 = [],[]
for idx in range(trackers.shape[0]):
    tracker = torch.tensor(trackers[idx]).unsqueeze(0).to(device).type(tensor_type)
    TRACKER = torch.tensor(TRACKERS[idx]).unsqueeze(0).to(device).type(tensor_type)
    outs1.append(net.decode(tracker).detach().cpu().numpy().squeeze())
    outs2.append(net.decode(TRACKER).detach().cpu().numpy().squeeze())

OUTS1 = np.hstack(np.array(outs1)) #2nd row
OUTS2 = np.hstack(np.array(outs2)) #3rd row

IMAGE = np.vstack(np.array([IMAGES, OUTS1, OUTS2])) #full image

#plot image
cmap = plt.cm.gray
norm = plt.Normalize(vmin=0, vmax=1)
#plt.imsave("/Users/cesaremagnetti/Documents/BEng_project/images_thesis/some_images.png",cmap(norm(IMAGE)))
fig = plt.figure(2)
plt.imshow(cmap(norm(IMAGE)))
plt.axis("off")
plt.show()

