from dataset.TrackerDS_advanced import TrackerDS_advanced
import engine_new as autoencoder_engine
from transforms import itk_transforms as itktransforms
from transforms import tensor_transforms as tensortransforms
from torchvision import transforms as torchtransforms
import numpy as np
import torch
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

#DATASET
train_ds = TrackerDS_advanced("/Users/cesaremagnetti/Documents/BEng_project/data/phantom_04Mar2020", N_samples = 10,
                              random_seed=42,mode = 'validate', transform=transform, target_transform = ConstantRescale)
test_ds = TrackerDS_advanced("/Users/cesaremagnetti/Documents/BEng_project/data/phantom_04Mar2020", N_samples = 5,
                             random_seed=42,mode = 'validate', transform=transform, target_transform = ConstantRescale)

train_ds.remove_corrupted_indexes(train_ds.get_corrupted_indexes())
test_ds.remove_corrupted_indexes(test_ds.get_corrupted_indexes())

np.random.seed(42)
test_idxs = np.random.randint(len(test_ds), size=(5,))
train_idxs = np.random.randint(len(train_ds), size=(5,))
 #try GAN finetuning
# checkpoint = torch.load("/Users/cesaremagnetti/projects/PyCharm/UltrasoundSimulator/models/DECODER/final_models/"
#                         "phantom/DECODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/resize_conv/last.tar", map_location='cpu')

checkpoint = None


model_root = "/Users/cesaremagnetti/projects/PyCharm/UltrasoundSimulator/models/AUTOENCODER/final_models/"\
             "phantom/AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/GAN_training/"
results_root = "/Users/cesaremagnetti/projects/PyCharm/UltrasoundSimulator/results/AUTOENCODER/final_models/" \
               "phantom/AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/GAN_training/"

autoencoder_engine.GAN_training(train_ds, test_ds, model_root, results_root, test_idxs, train_idxs, 0.0002, checkpoint=None, START_EPOCH = 0, MAX_EPOCH=200)