import torch
from matplotlib import pyplot as plt
from architectures.decoder import Generator
from dataset.TrackerDS_advanced import TrackerDS_advanced
import transforms.itk_transforms as itktransforms
import torchvision.transforms as torchtransforms
import transforms.tensor_transforms as tensortransforms
import numpy as np
import os

#TRANSFORMS
resample = itktransforms.Resample(new_spacing=[.5, .5, 1.])
tonumpy = itktransforms.ToNumpy(outputtype=np.float)
totensor = torchtransforms.ToTensor()
crop     = tensortransforms.CropToRatio(outputaspect=1.)
resize   = tensortransforms.Resize(size=(256,256), interp='BILINEAR')
rescale  = tensortransforms.Rescale(interval=(0,1))
transform = torchtransforms.Compose([resample,
                                     tonumpy,
                                     totensor,
                                     crop,
                                     resize,
                                     rescale])

ConstantRescale = tensortransforms.ConstantRescale(scaling = np.array([1/250,1/250,1/500,1.,1.,1.,1.]))

use_cuda = torch.cuda.is_available()


root = "/Users/cesaremagnetti/Desktop/GAN_models/DECODER/resize_conv/epoch150.tar"
checkpoint  =torch.load(root, map_location="cpu")
loss_D = checkpoint['loss_D']
loss_G = checkpoint['loss_G']
fig  = plt.figure()
plt.plot(range(len(loss_D)), loss_D, label = "Discriminator Loss")
plt.plot(range(len(loss_G)), loss_G, label = "Generator Loss")
plt.xlabel("iterations")
plt.legend()

