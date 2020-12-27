# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-

"""
script to generate a map of the tracker coordinates contained in the data according to the MSELoss that they generate with respect to an orginal sample.
a pretrain model is used to evaluate the loss of each point.
"""

# Author:
# Cesare Magnetti <cesare.magnetti98@gmail.com> <cesare.magnetti@kcl.ac.uk>
#    King's College London, UK
from architectures.autoencoder import Generator as Autoencoder
from architectures.decoder import Generator as Decoder
from architectures.variational_autoencoder import Generator as Variational_Autoencoder
import tensor_transforms  as tensortransforms
import itk_transforms as itktransforms
from torchvision import transforms as torchtransforms
from TrackerDS_advanced import TrackerDS_advanced
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
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
#DATASET AND DATALOADER
#data_root = "/home/cm19/BEng_project/data/patient_data/iFIND00366_for_simulator"
data_root = "/Users/cesaremagnetti/Documents/BEng_project/data/data/"
ds = TrackerDS_advanced(data_root, mode = "validate", transform=transform, target_transform = ConstantRescale)
ds1 = TrackerDS_advanced(data_root, mode = "validate", transform=transform,target_transform = ConstantRescale, X_point=(-30,-0,-350), X_radius=30)
print(len(ds), len(ds)-len(ds1))
ds.remove_corrupted_indexes(ds.get_corrupted_indexes())
ds1.remove_corrupted_indexes(ds1.get_corrupted_indexes())
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# roots = [ "/Users/cesaremagnetti/linode/models/DECODER/final_models/phantom/"
#           "DECODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/pretrained/resize_conv/last.tar",
#
#           "/Users/cesaremagnetti/linode/models/DECODER/final_models/phantom/incomplete_dataset/"
#           "DECODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/pretrained/resize_conv/last.tar",]

# roots = [ "/Users/cesaremagnetti/different_variationals/VARIATIONAL_AUTOENCODER/final_models/phantom/"
#           "VARIATIONAL_AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/BETA_1e-03/resize_conv/last.tar",
#
#           "/Users/cesaremagnetti/linode/models/VARIATIONAL_AUTOENCODER/final_models/phantom/incomplete_dataset/" \
#           "VARIATIONAL_AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/resize_conv/last.tar",]

# roots = [ "/Users/cesaremagnetti/linode/models/AUTOENCODER/final_models/phantom/"
#           "AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/resize_conv_old/last.tar",
#
#           "/Users/cesaremagnetti/linode/models/AUTOENCODER/final_models/phantom/incomplete_dataset/"
#           "AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/resize_conv/last.tar",]

roots = [ "/Users/cesaremagnetti/new_autoencoder/last.tar",

          "/Users/cesaremagnetti/new_autoencoder/incomplete/last.tar",]


names = ["trained on full dataset","trained on incomplete dataset"]
nets =  ()
for idx, (root) in enumerate(roots):

    checkpoint = torch.load(root, map_location='cpu')
    cfg = checkpoint['model_info']['model_configuration']
    for key in cfg.keys():
        print("{} --> {}".format(key,cfg[key]))

    if "DECODER" in cfg['net_name']:
        net = Decoder(cfg).to(device)
    elif "VARIATIONAL_AUTOENCODER" in cfg['net_name']:
        net = Variational_Autoencoder(cfg).to(device)
    elif "AUTOENCODER" in cfg['net_name']:
        net = Autoencoder(cfg).to(device)
        net.print_info(verbose=True)

    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    nets+=(net,)

x,y,z = [],[],[]
loss, loss1=[],[]
criterion = torch.nn.MSELoss()
for idx in range(len(ds)):
    img,tr = ds[idx]

    img = img.unsqueeze(0).type(torch.FloatTensor).to(device)
    x.append(tr[0])
    y.append(tr[1])
    z.append(tr[2])
    tr = torch.tensor(tr).unsqueeze(0).type(torch.FloatTensor).to(device)
    out = nets[0].decode(tr)
    out1 = nets[1].decode(tr)

    loss.append(criterion(out, img).item())
    loss1.append(criterion(out1, img).item())

    #print some info on where we at
    if idx%100 == 0:
        print("[{}]/[{}]".format(idx,len(ds)))

x1,y1,z1 = [],[],[]
for idx1 in range(len(ds1)):
    _,tr1 = ds1[idx1]

    x1.append(tr1[0])
    y1.append(tr1[1])
    z1.append(tr1[2])
loss,loss1 = np.array(loss), np.array(loss1)

MAX = np.amax([np.amax(loss), np.amax(loss1)])
#MAX = 0.06

hot_r = matplotlib.cm.get_cmap('hot_r')

# set up a figure thrice as wide as it is tall
fig1 = plt.figure(figsize=plt.figaspect(0.3))
#fig1.suptitle('{}'.format(decoders[0].get_info("model_name")), fontsize=10)
ax = fig1.add_subplot(1, 4, 1)
ax.scatter(y, z, c='b', alpha = 0.3, label = "full dataset")
ax.scatter(y1, z1, c='r', alpha = 0.3, label = "incomplete dataset")
plt.legend()
ax.title.set_text("datasets' samples")

ax = fig1.add_subplot(1, 4, 2)
img1 = ax.scatter(y, z, c=loss, alpha = 0.3, cmap = plt.hot(), vmin = 0., vmax = MAX)
fig1.colorbar(img1)
ax.title.set_text(names[0])

ax = fig1.add_subplot(1, 4, 3)
img1 = ax.scatter(y, z, c=loss1, alpha = 0.3, cmap = plt.hot(), vmin = 0., vmax = MAX)
fig1.colorbar(img1)
ax.title.set_text(names[1])

ax = fig1.add_subplot(1, 4, 4)
img1 = ax.scatter(y, z, c=np.divide(abs(loss1-loss),loss1)*100, alpha = 0.3, cmap = hot_r, vmin = 0, vmax = 100)
fig1.colorbar(img1)
ax.title.set_text("percentage difference")

plt.tight_layout()
plt.show()