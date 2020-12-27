import json
from architectures.decoder import Generator
import numpy as np
import os




cfg1 = {"mode": "resize_conv",
           "sizes": ((16, 16), (32, 32), (64, 64), (128, 128), (256, 256)),
           "nchannels": (32, 32, 32, 32, 32, 1),
           "kernel_size": (3,3,3,3,3),
           "stride": (1,1,1,1,1),
           "batchnorm2d": True,
           "dropout": False,
           "z_size": 7,
           "net_name": "DECODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1",
           "nlinear": (7, 32, 256, 512, 1024, 2048),
           "image_size": (8,8),
           #‘end_with_sigmoid’: True,
           }



cfgs = [cfg1,]
model_root = "/home/cm19/BEng_project/models/DECODER/DeeperDec/noDropout/MSELoss/try_different_models/visualize_gradients/"
results_root = "/home/cm19/BEng_project/results/DECODER/DeeperDec/noDropout/MSELoss/train_and_validation_images/" \
               "try_different_models/visualize_gradients/"

for cfg in cfgs:
    net = Generator(cfg)
    net.print_info(verbose=True)

with open(model_root + "configurations.txt", "w") as f:
    f.write(json.dumps(cfgs))

with open(model_root + "configurations.txt", "r") as f:
    cfgs = json.load(f)

def tuplify(thing):
    if isinstance(thing, list): return tuple(map(tuplify, thing))
    else: return thing

for cfg in cfgs:
    for key in cfg:
        cfg[key] = tuplify(cfg[key])
    net = Generator(cfg)
    print(net.get_info('model_name'))

for cfg in cfgs:
        os.mkdir(model_root + cfg['net_name'])
        os.mkdir(model_root+cfg['net_name']+"/resize_conv")
        os.mkdir(results_root + cfg['net_name'])
        os.mkdir(results_root+cfg['net_name']+"/resize_conv")
        os.mkdir(results_root + cfg['net_name'] + "/resize_conv/train")
        os.mkdir(results_root + cfg['net_name'] + "/resize_conv/validate")
        #os.rmdir(results_root + cfg['net_name'] + "/resize_conv/validate")
        #os.rmdir(results_root + cfg['net_name'] + "/resize_conv/train")
        #os.rmdir(results_root+cfg['net_name']+"/resize_conv")
        #os.rmdir(results_root + cfg['net_name'])
        #os.rmdir(model_root+cfg['net_name']+"/resize_conv")
        #os.rmdir(model_root+cfg['net_name'])







