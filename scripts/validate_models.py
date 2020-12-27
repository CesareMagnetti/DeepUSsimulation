# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""
script to validate models given a criterion
"""
# Author:
# Cesare Magnetti <cesare.magnetti98@gmail.com> <cesare.magnetti@kcl.ac.uk>
#    King's College London, UK

from architectures.autoencoder import Generator as Autoencoder
from architectures.decoder import Generator as Decoder
from dataset.TrackerDS_advanced import TrackerDS_advanced
from engine import *
from transforms import itk_transforms as itktransforms
from transforms import tensor_transforms as tensortransforms
from torchvision import transforms as torchtransforms
import torch
import torch.nn as nn
import losses.SSIMLoss as SSIMLoss
import losses.utils as utils
from torch.backends import cudnn
import timeit
#TRANSFORMS
resample = itktransforms.Resample(new_spacing=[.5, .5, 1.])
tonumpy  = itktransforms.ToNumpy(outputtype='float')
totensor = torchtransforms.ToTensor()
crop     = tensortransforms.CropToRatio(outputaspect=1.)
resize   = tensortransforms.Resize(size=(256,256), interp='BILINEAR')
rescale  = tensortransforms.Rescale(interval=(0,1))
transform = torchtransforms.Compose([resample, tonumpy, totensor, crop, resize, rescale])
ConstantRescale = tensortransforms.ConstantRescale(scaling = np.array([1/250,1/250,1/500,1.,1.,1.,1.]))#DATASET AND DATALOADER

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")#DEFINE ROOT DIRECTORY OF EACH MODEL
roots = [
         "/home/cm19/BEng_project/models/DECODER/DeeperDec/noDropout/MSELoss/bigger_latent_image/DECODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/deconv/last.tar",

         "/home/cm19/BEng_project/models/AUTOENCODER/DeeperAuto/noDropout/MSELoss/bigger_latent_image/AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/deconv/last.tar",

         "/home/cm19/BEng_project/models/AUTOENCODER/DeeperAuto/noDropout/MSELoss/bigger_latent_image/AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/" \
         "refined_freezing_encoder/deconv/last.tar",
        ]

names = ["decoder","autoencoder\nnormal","pretrained\nautoencoder\nfinetuned"]
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
    elif "AUTOENCODER" in cfg['net_name']:
        idxs.append(idx)
        net = Autoencoder(cfg).to(device)
    else:
        net = Decoder(cfg).to(device)
        print("WARNING: inconsistent name for root directory:\n{}\n, decoder architecture was instanciated by default,"\
              " rename the root directory accordingly if needed:\n i.e. ~/ANY PATH/ARCHITECTURE TYPE (egs: DECODER or"\
              " AUTOENCODER)/ANY PATH/model.tar")    # 1. filter out unnecessary keys
    pretrained_dict = {}
    flag = True
    for k, v in checkpoint['model_state_dict'].items():
        if k not in net.state_dict():
            if flag:
                print("the following parameters will be discarded:\n")
                flag = False
            print("{}: {}".format(k, v))
        else:
            pretrained_dict[k] = v    #load state dict
    net.load_state_dict(pretrained_dict)    #net.print_info(True)
    net.eval()
    #names.append(net.get_info('model_name'))
    nets+=(net,)

root = "/home/cm19/BEng_project/data/patient_data/iFIND00366_for_simulator"
root = "/home/cm19/BEng_project/data/data"
test_ds = TrackerDS_advanced(root, mode = 'validate', transform=transform, target_transform = ConstantRescale)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, num_workers=0, shuffle = False)
test_loader.pin_memory = True
criterion = utils.PSNR()
cudnn.benchmark = True
#criterion.cuda
for net in nets:
    running_loss = 0
    running_loss1 = 0
    for batch_idx, (img, target) in enumerate(test_loader):
        img = img.to(device)
        target = target.to(device)
        img = img.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.FloatTensor)
        out = net.decode(target)
        try:
            out1,_ = net(img)
        except:
            out1 = net(target)
        loss = criterion(out, img)
        running_loss += loss.item()
        loss1 = criterion(out1, img)
        running_loss1 += loss1.item()
    print("validation_loss = {}\t|\t{}".format(running_loss/len(test_loader), running_loss1/len(test_loader)))

