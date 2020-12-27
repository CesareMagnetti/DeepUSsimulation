# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-

"""
script to merge two losses arrays from checkpoints
"""

# Author:
# Cesare Magnetti <cesare.magnetti98@gmail.com> <cesare.magnetti@kcl.ac.uk>
# 	  King's College London, UK

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


#DEFINE ROOT DIRECTORY OF EACH MODEL
root1 = "/home/cm19/BEng_project/models/DECODER/DeeperDec/noDropout/MSELoss/bigger_latent_image/DECODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/deconv/epoch_120.tar"
root2 = "/home/cm19/BEng_project/models/DECODER/DeeperDec/noDropout/MSELoss/bigger_latent_image/DECODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/deconv/last.tar"
model_root = "/home/cm19/BEng_project/models/DECODER/DeeperDec/noDropout/MSELoss/bigger_latent_image/DECODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/deconv/last.tar"
loss1,loss2 = {},{}
checkpoint = torch.load(root1, map_location='cpu')
loss1['train'] = checkpoint['train_loss_hist']
loss1['validate'] = checkpoint['validation_loss_hist']
checkpoint = torch.load(root2, map_location='cpu')
loss2['train'] = checkpoint['train_loss_hist']
loss2['validate'] = checkpoint['validation_loss_hist']

losses = {}
for key1, key2 in zip(loss1,loss2):
    assert key1==key2, "wrong losses trying to be merged, check loaded models"
    losses[key1] = np.concatenate((loss1[key1], loss2[key2]))


checkpoint = torch.load(model_root, map_location='cpu')
savepath = "/home/cm19/BEng_project/models/DECODER/DeeperDec/noDropout/MSELoss/bigger_latent_image/DECODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/deconv/last_copy.tar"
torch.save({'model_state_dict': checkpoint['model_state_dict'],
            'validation_loss_hist': losses['validate'],
            'train_loss_hist': losses['train'],
            'model_info': checkpoint['model_info'],
            }, savepath)