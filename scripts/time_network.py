# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-

"""
script to generate a simulated trajectory, starting from a validation tracker
"""

# Author:
# Cesare Magnetti <cesare.magnetti98@gmail.com> <cesare.magnetti@kcl.ac.uk>
# 	  King's College London, UK

from architectures.autoencoder import Generator as Autoencoder
from architectures.decoder import Generator as Decoder
from architectures.variational_autoencoder import Generator as Variational_Autoencoder
from dataset.TrackerDS_advanced import TrackerDS_advanced
from engine import *
from transforms import itk_transforms as itktransforms
from transforms import tensor_transforms as tensortransforms
from torchvision import transforms as torchtransforms
import torch
from time import process_time
import glob
from matplotlib import pyplot as plt



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
ds = TrackerDS_advanced(root, mode = "validate", transform=transform, target_transform=ConstantRescale)#,reject_inside_radius=False, X_point=(-30,-0,-350), X_radius=30)
# root = "/home/cm19/BEng_project/data/phantom_04Mar2020"
# ds = TrackerDS_advanced(root, mode = "validate", transform=transform, target_transform=ConstantRescale)

#CUDA SETUP
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
tensortype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

#load networks###############################################
roots = [
    "/users/cesaremagnetti/final_models/phantom/DECODER/last.tar",
    "/users/cesaremagnetti/final_models/phantom/PRETRAINED/last.tar",
    "/users/cesaremagnetti/final_models/phantom/AUTOENCODER/last.tar",
    "/users/cesaremagnetti/final_models/phantom/VARIATIONAL_AUTOENCODER/last.tar"
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

_, tracker = ds[np.random.randint(len(ds))]
tracker = torch.tensor(tracker).unsqueeze(0).to(device).type(tensortype)
time = []
for net in nets:
    print(net.get_info("model_name"))
    for j in range(20):
        print(j+1)
        start = process_time()
        for i in range(500):
            out = net.decode(tracker)
        end = process_time()
        time.append((end-start)/500)

    #print mean and std
    print(np.mean(time), np.std(time))