#import python modules
import os
import sys
import numpy as np
import torch
from torchvision import transforms as torchtransforms

# #append current working directory to the path
# conf_path = os.getcwd()
# sys.path.append(conf_path)

#import custom modules
from ..architectures.autoencoder import Generator
from ..dataset.TrackerDS_advanced import TrackerDS_advanced
from ..scripts import engine_new as autoencoder_engine
#import scripts.engine_variational as variational_engine
from ..transforms import itk_transforms as itktransforms
from ..transforms import tensor_transforms as tensortransforms



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

MODEL_ROOT = "~/linode/models/AUTOENCODER/final_models/" if use_cuda\
             else "/Users/cesaremagnetti/linode/models/AUTOENCODER/final_models/"
RESULTS_ROOT = "~/linode/results/AUTOENCODER/final_models/" if use_cuda\
             else "/Users/cesaremagnetti/linode/results/AUTOENCODER/final_models/"


ds_root = "~/linode/data/phantom_04Mar2020" if use_cuda\
          else "/Users/cesaremagnetti/Documents/BEng_project/data/phantom_04Mar2020"
if "phantom" in ds_root:
    MODEL_ROOT+="phantom/"
    RESULTS_ROOT+="phantom/"
elif "patient" in ds_root:
    MODEL_ROOT+="real_patient/"
    RESULTS_ROOT+="real_patient/"


#create the configurations#####################################
cfg1 = {"mode": "resize_conv",
        "sizes": ((16, 16), (32, 32), (64, 64), (128, 128), (256, 256)),
        "nchannels": (32,32,32,32,32,1),
        "kernel_size": (4,4,4,4,4),
        "stride": (2,2,2,2,2),
        "batchnorm2d": True,
        "dropout": False,
        "z_size": 7,
        "net_name": "AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1",
        "nlinear": (7, 32, 256, 512, 1024, 2048),
        "image_size": (8, 8),
        "k2": 0e+00
           }

cfg2 = {"mode": "resize_conv",
        "sizes": ((16, 16), (32, 32), (64, 64), (128, 128), (256, 256)),
        "nchannels": (32,32,32,32,32,1),
        "kernel_size": (4,4,4,4,4),
        "stride": (2,2,2,2,2),
        "batchnorm2d": True,
        "dropout": False,
        "z_size": 7,
        "net_name": "AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1",
        "nlinear": (7, 32, 256, 512, 1024, 2048),
        "image_size": (8, 8),
        "k2": 2e-01
           }

cfg3 = {"mode": "resize_conv",
        "sizes": ((16, 16), (32, 32), (64, 64), (128, 128), (256, 256)),
        "nchannels": (32,32,32,32,32,1),
        "kernel_size": (4,4,4,4,4),
        "stride": (2,2,2,2,2),
        "batchnorm2d": True,
        "dropout": False,
        "z_size": 7,
        "net_name": "AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1",
        "nlinear": (7, 32, 256, 512, 1024, 2048),
        "image_size": (8, 8),
        "k2": 4e-01
           }

cfg4 = {"mode": "resize_conv",
        "sizes": ((16, 16), (32, 32), (64, 64), (128, 128), (256, 256)),
        "nchannels": (32,32,32,32,32,1),
        "kernel_size": (4,4,4,4,4),
        "stride": (2,2,2,2,2),
        "batchnorm2d": True,
        "dropout": False,
        "z_size": 7,
        "net_name": "AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1",
        "nlinear": (7, 32, 256, 512, 1024, 2048),
        "image_size": (8, 8),
        "k2": 6e-01
           }

cfg5 = {"mode": "resize_conv",
        "sizes": ((16, 16), (32, 32), (64, 64), (128, 128), (256, 256)),
        "nchannels": (32,32,32,32,32,1),
        "kernel_size": (4,4,4,4,4),
        "stride": (2,2,2,2,2),
        "batchnorm2d": True,
        "dropout": False,
        "z_size": 7,
        "net_name": "AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1",
        "nlinear": (7, 32, 256, 512, 1024, 2048),
        "image_size": (8, 8),
        "k2": 8e-01
           }

cfgs = [cfg1,cfg2,cfg3,cfg4,cfg5]
#create the model roots and results roots for the models we will train
model_roots = []
results_roots = []
for cfg in cfgs:
    if not os.path.exists(MODEL_ROOT + cfg['net_name']):
        os.mkdir(MODEL_ROOT + cfg['net_name'])
    if not os.path.exists(MODEL_ROOT + cfg['net_name'] + "/k2_{:.0e}".format(cfg['k2'])):
        os.mkdir(MODEL_ROOT + cfg['net_name'] + "/k2_{:.0e}".format(cfg['k2']))
    if not os.path.exists(MODEL_ROOT + cfg['net_name'] + "/k2_{:.0e}".format(cfg['k2'])+ "/" + cfg['mode']):
            os.mkdir(MODEL_ROOT + cfg['net_name'] + "/k2_{:.0e}".format(cfg['k2'])+ "/" + cfg['mode'])
    if not os.path.exists(RESULTS_ROOT + cfg['net_name']):
        os.mkdir(RESULTS_ROOT + cfg['net_name'])
    if not os.path.exists(RESULTS_ROOT + cfg['net_name']+ "/k2_{:.0e}".format(cfg['k2'])):
        os.mkdir(RESULTS_ROOT + cfg['net_name']+ "/k2_{:.0e}".format(cfg['k2']))
    if not os.path.exists(RESULTS_ROOT + cfg['net_name']+ "/k2_{:.0e}".format(cfg['k2'])+ "/" + cfg['mode']):
        os.mkdir(RESULTS_ROOT + cfg['net_name']+ "/k2_{:.0e}".format(cfg['k2'])+ "/" + cfg['mode'])
    if not os.path.exists(RESULTS_ROOT + cfg['net_name']+ "/k2_{:.0e}".format(cfg['k2'])+ "/" + cfg['mode'] + "/train"):
        os.mkdir(RESULTS_ROOT + cfg['net_name']+ "/k2_{:.0e}".format(cfg['k2'])+ "/" + cfg['mode'] + "/train")
    if not os.path.exists(RESULTS_ROOT + cfg['net_name']+ "/k2_{:.0e}".format(cfg['k2'])+ "/" + cfg['mode'] + "/validate"):
        os.mkdir(RESULTS_ROOT + cfg['net_name'] + "/k2_{:.0e}".format(cfg['k2']) + "/" + cfg['mode'] + "/validate")


    model_roots.append(MODEL_ROOT + cfg['net_name'] + "/k2_{:.0e}/".format(cfg['k2']))
    results_roots.append(RESULTS_ROOT + cfg['net_name'] + "/k2_{:.0e}/".format(cfg['k2']))
########################################################################

#create datasets###########################################################
train_ds = TrackerDS_advanced(ds_root, mode = 'train', N_samples=10, transform=transform, target_transform = ConstantRescale)
test_ds = TrackerDS_advanced(ds_root, mode = 'validate',N_samples=10, transform=transform, target_transform = ConstantRescale)
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
    net.print_info(verbose=True)
    if os.path.isfile(model_roots[idx]+"/" +cfg['mode'] + "/last_copy.tar"):
        checkpoint = torch.load(model_roots[idx]+"/" +cfg['mode'] + "/last_copy.tar", map_location = "cpu")
    else:
        checkpoint = None
    autoencoder_engine.train_session(net, train_ds, test_ds, model_roots[idx], results_roots[idx],checkpoint, test_idxs,
                                     train_idxs, MAX_EPOCH=2, lr=0.0002)
