from architectures.decoder import Generator
from dataset.TrackerDS_advanced import TrackerDS_advanced
import scripts.decoder_engine as decoder_engine
from transforms import itk_transforms as itktransforms
from transforms import tensor_transforms as tensortransforms
from torchvision import transforms as torchtransforms
import numpy as np
import json
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

MODEL_ROOT = "/home/cm19/BEng_project/models/DECODER/DeeperDec/noDropout/MSELoss/try_different_models/visualize_gradients/"
RESULTS_ROOT = "/home/cm19/BEng_project/results/DECODER/DeeperDec/noDropout/MSELoss/train_and_validation_images/" \
               "try_different_models/visualize_gradients/"

phantom_root = "/home/cm19/BEng_project/data/phantom_04Mar2020"
#read configurations from text file##########################
with open(MODEL_ROOT + "configurations.txt", "r") as f:
    cfgs = json.load(f)

#json converts tuples to lists, we don't want that, the following code restores the tuples
def tuplify(thing):
    if isinstance(thing, list): return tuple(map(tuplify, thing))
    else: return thing

for cfg in cfgs:
    for key in cfg:
        cfg[key] = tuplify(cfg[key])
##################################################################

#create the model roots and results roots for the models we will train
model_roots = []
results_roots = []
for cfg in cfgs:
    print(cfg['net_name'])
    model_roots.append(MODEL_ROOT + cfg['net_name'] + "/")
    results_roots.append(RESULTS_ROOT + cfg['net_name'] + "/")
########################################################################

#create datasets###########################################################
train_ds = TrackerDS_advanced(phantom_root, mode = 'train', transform=transform, target_transform = ConstantRescale)
test_ds = TrackerDS_advanced(phantom_root, mode = 'validate', transform=transform, target_transform = ConstantRescale)
print("datasets before filtering corrupted files:\n train set: {} samples\n test set: {} samples\n".format(len(train_ds),len(test_ds)))
train_ds.remove_corrupted_indexes(train_ds.get_corrupted_indexes())
test_ds.remove_corrupted_indexes(test_ds.get_corrupted_indexes())
print("datasets after filtering corrupted files:\n train set: {} samples\n test set: {} samples\n".format(len(train_ds),len(test_ds)))
#############################################################################

# initiate indices to sample at each epoch
np.random.seed(42)
test_idxs = np.random.randint(len(test_ds), size=(5,))
train_idxs = np.random.randint(len(train_ds), size=(5,))

#launch training session for each model############################################
for idx in range(len(cfgs)):
    net = Generator(cfgs[idx])
    decoder_engine.train_session(net, train_ds, test_ds, model_roots[idx], results_roots[idx], test_idxs, train_idxs, MAX_EPOCH=200,
                                 lr=0.0002)
