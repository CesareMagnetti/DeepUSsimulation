from architectures.autoencoder import Generator
from losses.MaskedMSELoss import MaskedMSELoss
from dataset.TrackerDS_advanced import TrackerDS_advanced
from engine import *
from transforms import itk_transforms as itktransforms
from transforms import tensor_transforms as tensortransforms
from torchvision import transforms as torchtransforms
import numpy as np
import torch
import torch.nn as nn
import glob
from torch.backends import cudnn
import sys

#TRANSFORMS
tonumpy = itktransforms.ToNumpy(outputtype=np.float)
totensor = torchtransforms.ToTensor()
crop     = tensortransforms.CropToRatio(outputaspect=1.)
resize   = tensortransforms.Resize(size=(256,256), interp='BILINEAR')
rescale  = tensortransforms.Rescale(interval=(0,1))
ConstantRescale = tensortransforms.ConstantRescale(scaling = np.array([1/250,1/250,1/500,1.,1.,1.,1.]))
ConstantRescale1 = tensortransforms.ConstantRescale(scaling = np.array([250,250,500,1.,1.,1.,1.]))
transform = torchtransforms.Compose([tonumpy,
                                     totensor,
                                     crop,
                                     resize,
                                     rescale])
target_transform = torchtransforms.Compose([ConstantRescale, ConstantRescale1])

test_ds1 = TrackerDS_advanced("/home/cm19/BEng_project/data/data", mode = 'train', transform=transform, target_transform = None)
test_ds2 = TrackerDS_advanced("/home/cm19/BEng_project/data/data", mode = 'train', transform=transform, target_transform = target_transform)

for s1,s2 in zip(test_ds1,test_ds2):
    _,t1 = s1
    _,t2 = s2
    print(t1)
    print(t2)