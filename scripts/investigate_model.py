from architectures.DeeperAuto import Generator
from losses.MaskedMSELoss import MaskedMSELoss
from dataset.TrackerDS import TrackerDS
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
resample = itktransforms.Resample(new_spacing=[.5, .5, 1.])
tonumpy  = itktransforms.ToNumpy(outputtype='float')
totensor = torchtransforms.ToTensor()
crop     = tensortransforms.CropToRatio(outputaspect=1.)
resize   = tensortransforms.Resize(size=(256,256), interp='BILINEAR')
rescale  = tensortransforms.Rescale(interval=(0,1))
transform = torchtransforms.Compose([resample, tonumpy, totensor, crop, resize, rescale])

#DATASET AND DATALOADER
root = "/home/cm19/BEng_project/data/data"
ds = TrackerDS(root, mode = "train", transform=transform)

#CUDA SETUP
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#DEFINE ROOT DIRECTORY AND MODALITIES (deconv and resize_conv)
#cfgs = (config,)

model_roots = ["/home/cm19/BEng_project/models/DECODER/DeeperDec/noDropout/MSELoss/new_models/DEBUGGING/",]

results_roots = ["/home/cm19/BEng_project/results/DECODER/DeeperDec/noDropout/MSELoss/train_and_validation_images/new_models/DEBUGGING/",]


def find_image_sizes2D(image_size, n_channels, stride, ks):
    '''
    function to automatically detect image sizes of a general network for 2D images
    :param image_size: initial image size
    :param n_channels: successive n_channels of the model layers
    :param stride: array of strides of the successive layers
    :param ks: array of kernel sizes of the successive layers
    :param j: recursion flag

    :return: tuple containing all the sizes of each layer
    '''

    padding = tuple(np.int((k - 1) / 2) for k in ks)
    def rec(s, n_channels, stride, ks, padding, j=0):
        '''
        recursion function to get successive sizes
        '''
        if j < len(n_channels) - 1:
            #conv2d
            #new_size = [np.floor((i + 2 * padding[j] - (ks[j] - 1) - 1) / stride[j] + 1) for i in s]
            #convtranspose2d
            new_size = [(i-1)*stride[j] - 2*padding[j] + ks[j] for i in s]
            s = s + rec(tuple(new_size), n_channels, stride, ks, padding, j=j + 1)
        return s
    # get image sizes of successive layers
    image_size = rec(image_size, n_channels, stride, ks, padding)
    # reshape to format: ((l1,w1),(l2,w2)...)
    image_size = [(image_size[i], image_size[i + 1]) for i in range(0, len(image_size), 2)]
    return tuple(image_size)


