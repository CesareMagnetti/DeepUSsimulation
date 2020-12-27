# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-

"""
script to generate a simulated trajectory, starting from a validation tracker. taking a random sample
from validation set and slightly perturbate it to see what happens
"""

# Author:
# Cesare Magnetti <cesare.magnetti98@gmail.com> <cesare.magnetti@kcl.ac.uk>
# 	  King's College London, UK

from architectures.variational_autoencoder import Generator as Variational_Autoencoder
from architectures.autoencoder import Generator as Autoencoder
from architectures.decoder import Generator as Decoder
from dataset.TrackerDS_advanced import TrackerDS_advanced
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

#DATASET AND DATALOADER
root = "/home/cm19/BEng_project/data/patient_data/iFIND00366_for_simulator"
ds = TrackerDS_advanced(root, mode = "validate", transform=transform, target_transform=ConstantRescale)

#CUDA SETUP
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#DEFINE ROOT DIRECTORY OF EACH MODEL
roots = [
         "/home/cm19/BEng_project/models/DECODER/DeeperDec/noDropout/MSELoss/bigger_latent_image/"\
         "DECODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/real_patient_data/deconv/last.tar",

         "/home/cm19/BEng_project/models/AUTOENCODER/DeeperAuto/noDropout/MSELoss/bigger_latent_image/"\
         "AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/real_patient_data/deconv/last.tar",

         "/home/cm19/BEng_project/models/VARIATIONAL_AUTOENCODER/DeeperAutoVar/noDropout/MSELoss/" \
         "VARIATIONAL_AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/real_patient/deconv/last.tar",

         "/home/cm19/BEng_project/models/AUTOENCODER/DeeperAuto/noDropout/MSELoss/bigger_latent_image/"\
         "AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/real_patient_data/refined_freezing_encoder/deconv/last.tar",
        ]

names = ["decoder","autoencoder","variational\nautoencoder","pretrained\ndecoder"]
#names.append("guess 2")
names = np.array(names)
#names = []
nets = ()
idxs = []
for idx, root in enumerate(roots):
    checkpoint = torch.load(root, map_location='cpu')
    cfg = checkpoint['model_info']['model_configuration']
    for key in cfg.keys():
        print("{} --> {}".format(key, cfg[key]))
    if "DECODER" in cfg['net_name']:
        net = Decoder(cfg).to(device)
    elif "VARIATIONAL_AUTOENCODER" in cfg['net_name']:
        idxs.append(idx)
        net = Variational_Autoencoder(cfg).to(device)
    elif "AUTOENCODER" in cfg['net_name']:
        idxs.append(idx)
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

    #load state dict
    net.load_state_dict(pretrained_dict)

    #net.print_info(True)
    net.eval()
    #names.append(net.get_info('model_name'))
    nets+=(net,)

#create the trajectory
TRACKERS = []
N_steps = 10
image1, tracker1 = ds[np.random.randint(len(ds))]
image2, tracker2 = ds[np.random.randint(len(ds))]
diff = (tracker2-tracker1)/N_steps
TRACKERS.append(tracker1)
for i in range(N_steps):
    #add i to the z component (scaled accordingly!) for example
    new_tracker = tracker1 + (i+1)*diff
    TRACKERS.append(new_tracker)
TRACKERS.append(tracker2)

#generate full image
OUTS = []
for net in nets:
    outs = []
    for tracker in TRACKERS:
        tracker = torch.tensor(tracker).unsqueeze(0).to(device).type(torch.cuda.FloatTensor)
        outs.append(net.decode(tracker).detach().cpu().numpy().squeeze())
    OUTS.append(np.hstack(np.array([image1.detach().cpu().numpy().squeeze(),] + outs + [image2.detach().cpu().numpy().squeeze(),])))

IMAGE = np.vstack(np.array(OUTS))

#plot image
cmap = plt.cm.gray
norm = plt.Normalize(vmin=0, vmax=1)
#plt.imsave("/Users/cesaremagnetti/Documents/BEng_project/images_thesis/some_images.png",cmap(norm(IMAGE)))
fig = plt.figure()
plt.imshow(cmap(norm(IMAGE)))
plt.axis("off")
plt.show()