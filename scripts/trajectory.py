# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-

"""
script to generate a simulated trajectory, starting from a validation tracker
"""

# Author:
# Cesare Magnetti <cesare.magnetti98@gmail.com> <cesare.magnetti@kcl.ac.uk>
# 	  King's College London, UK

from architectures.variational_autoencoder import Generator as Variational_Autoencoder
from architectures.autoencoder import Generator as Autoencoder
from architectures.decoder import Generator as Decoder
from dataset.TrackerDS_advanced import TrackerDS_advanced
from engine import *
from transforms import itk_transforms as itktransforms
from transforms import tensor_transforms as tensortransforms
from torchvision import transforms as torchtransforms
import torch
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
phantom_root = "/home/cm19/BEng_project/data/data"
patient_root_extra = "/home/cm19/BEng_project/data/phantom_04Mar2020"
patient_root = "/home/cm19/BEng_project/data/patient_data/iFIND00366_for_simulator/"
# root = "/home/cm19/BEng_project/data/phantom_04Mar2020"
ds = TrackerDS_advanced(patient_root, mode = "validate", transform=transform, target_transform=ConstantRescale)

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

names = ["decoder",
         "autoencoder",
         "variational\nautoencoder",
         "pretrained\ndecoder"]
#names.append("guess 2")
names = np.array(names)
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

plt.ion()
plt.show()
fig, axes = plt.subplots(1, len(nets)+ 1)
fig.suptitle('DECODED TRAJECTORY', fontsize=10)
if len(idxs)>0:
    fig1, axes1 = plt.subplots(1, len(idxs)+ 1)
    fig1.suptitle('AUTOENCODED TRAJECTORY', fontsize=10)

for image,tracker in ds:
    #print(sum(tracker[3:]))
    tracker = torch.tensor(tracker).unsqueeze(0).type(torch.FloatTensor).to(device)
    image = image.unsqueeze(0).type(torch.FloatTensor).to(device)
    outs,outs1 = [],[]

    sigma = 4
    sigma_sim = 2

    for net in nets:
        out = net.decode(tracker)
        out = out.detach().cpu().numpy().squeeze()
        outs.append(out)
        #outs.append(gaussian_filter(out, sigma=sigma_sim))
        if "VARIATIONAL_AUTOENCODER" in net.get_info("model_name"):
            out1,_,_,_ = net(image)
            out1 = out1.detach().cpu().numpy().squeeze()
            outs1.append(out1)
        elif "AUTOENCODER" in net.get_info("model_name"):
            out1,_ = net(image)
            out1 = out1.detach().cpu().numpy().squeeze()
            outs1.append(out1)

    #if squared difference between two outputs is needed uncomment next line
    #outs.append((outs[0] - outs[1])**2)
    #if blurred input image is needed uncomment next line
    #outs.append(gaussian_filter(image.detach().cpu().numpy().squeeze(), sigma=sigma))
    axes[0].imshow(image.detach().cpu().numpy().squeeze(), interpolation='nearest', cmap = 'gray')
    axes[0].axis('off')
    axes[0].title.set_text('original image')
    for ax, name, out in zip(axes[1:], names, outs):
        ax.imshow(out, interpolation='nearest', cmap = 'gray')
        ax.axis('off')
        ax.title.set_text(name)
    if len(idxs) > 0:
        axes1[0].imshow(image.detach().cpu().numpy().squeeze(), interpolation='nearest', cmap='gray')
        axes1[0].axis('off')
        axes1[0].title.set_text('original image')
        for ax, name, out in zip(axes1[1:], names[idxs], outs1):
            ax.imshow(out, interpolation='nearest', cmap='gray')
            ax.axis('off')
            ax.title.set_text(name)
    plt.show()
    plt.pause(0.5)
    # matplotlib.pyplot.imsave(save_root + str(i)+".png",
    #                          out, cmap='gray')

