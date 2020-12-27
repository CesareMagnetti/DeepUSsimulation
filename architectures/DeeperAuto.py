import torch.nn as nn
import numpy as np
import sys
from torch.nn.modules.upsampling import UpsamplingNearest2d as UP
import torch.nn.utils.weight_norm as weightNorm

class View(nn.Module):
    """Simple module to `View` a tensor in a certain size

    Wrapper of the `torch.Tensor.View()` function into a forward module
    Args:
        size (tuple) output tensor size
    """

    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        # model attributes
        info = {}
        assert 'mode' in cfg, "ERROR: missing mandatory key in cfg: (key: <mode>)"
        assert 'net_name' in cfg, "ERROR: missing mandatory key in cfg: (key: <net_name>)"
        info['model_name'] = cfg['mode'] + "__" + cfg['net_name']
        info['model_configuration'] = cfg
        self.info = info
        nchannels =(256, 128, 128, 128, 64, 64, 64, 32, 32, 32, 16, 16, 16, 8, 8, 8, 1, 1, 1)
        ks = (4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 3, 3)
        strides = (2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1)
        if cfg['mode'] == 'resize_conv':
            sizes = ((8, 8), (8, 8), (8, 8), (16, 16), (16, 16), (16, 16), (32, 32), (32, 32),
                     (32, 32), (64, 64), (64, 64),(64, 64), (128, 128), (128, 128), (128, 128),
                     (256, 256), (256, 256), (256, 256))

        padding = tuple(np.int((k - 1) / 2) for k in ks)
        z_size = 7
        nlinear = (7, 512, 4096, 4096)

        unflatten = (nchannels[0],) + (4,4)
        flatten = np.prod(unflatten)

        if 'batchnorm2d' in cfg:
            batchnorm2d = cfg['batchnorm2d']
        else:
            batchnorm2d = False
        if 'dropout' in cfg:
            dropout = True
            conv_drop_prob = cfg['dropout']['conv_drop_prob']
            linear_drop_prob = cfg['dropout']['linear_drop_prob']
        else:
            dropout = False

        if 'weight normalization' in cfg:
            weight_norm = cfg['weight normalization']
        else:
            weight_norm = False

        dec = []
        enc = []

        # ENCODING LAYERS# # # # # # # # # #
        #convolutional layers
        for idx, (c1, c2, stride, k, pad) in enumerate(zip(nchannels[::-1][:-1], nchannels[::-1][1:], strides[::-1],
                                                                ks[::-1],padding[::-1])):
            if weight_norm:
                enc.append(weightNorm(nn.Conv2d(c1, c2, stride=stride, kernel_size=k, padding=pad),name = "weight"))
            else:
                enc.append(nn.Conv2d(c1, c2, stride=stride, kernel_size=k, padding=pad))

            if dropout:
                enc.append(nn.Dropout2d(conv_drop_prob))
            if batchnorm2d:
                enc.append(nn.BatchNorm2d(c2))
            if idx < len(nchannels[::-1][:-1]) - 1:
                enc.append(nn.ReLU(True))
        # view
        enc.append(View((-1, flatten)))
        #fully connected layers
        for idx, (c1, c2) in enumerate(zip(nlinear[::-1][:-1], nlinear[::-1][1:])):
            if weight_norm:
                enc.append(weightNorm(nn.Linear(c1, c2),name = "weight"))
            else:
                enc.append(nn.Linear(c1, c2))
            if dropout and idx > 0:
                enc.append(nn.Dropout(linear_drop_prob))
            if idx < len(nlinear[:-1]) - 1:
                enc.append(nn.ReLU(True))

        # DECODING LAYERS # # # # # # # # # #

        # fully connected layers
        for idx, (c1, c2) in enumerate(zip(nlinear[:-1], nlinear[1:])):
            if weight_norm:
                dec.append(weightNorm(nn.Linear(c1, c2),name = "weight"))
            else:
                dec.append(nn.Linear(c1, c2))
            if dropout and idx > 0:
                dec.append(nn.Dropout(linear_drop_prob))
            if idx < len(nlinear[:-1]) - 1:
                dec.append(nn.ReLU(True))

        # view
        dec.append(View((-1,) + unflatten))

        #conv layers
        if cfg['mode'] == "deconv":
            # conv layers
            for idx, (c1, c2, stride, k, pad) in enumerate(zip(nchannels[:-1], nchannels[1:], strides, ks, padding)):
                if weight_norm:
                    dec.append(weightNorm(nn.ConvTranspose2d(c1, c2, stride=stride, kernel_size=k, padding=pad), name = "weight"))
                else:
                    dec.append(nn.ConvTranspose2d(c1, c2, stride=stride, kernel_size=k, padding=pad))

                if dropout:
                    dec.append(nn.Dropout2d(conv_drop_prob))
                if batchnorm2d:
                    dec.append(nn.BatchNorm2d(c2))
                if idx < len(nchannels[:-1]) - 1:
                    dec.append(nn.ReLU(True))

        elif cfg['mode'] == "resize_conv":
            #FIXME: right now the upsampling convolutions will always be ks=3, stride = 1, padding = 1
            for idx, (c1, c2, size) in enumerate(zip(nchannels[:-1], nchannels[1:], sizes)):
                dec.append(UP(size))
                if weight_norm:
                    dec.append(weightNorm(nn.Conv2d(c1, c2, stride=1, kernel_size=3, padding=1),name = "weight"))
                else:
                    dec.append(nn.Conv2d(c1, c2, stride=1, kernel_size=3, padding=1))
                if dropout:
                    dec.append(nn.Dropout2d(conv_drop_prob))
                if batchnorm2d:
                    dec.append(nn.BatchNorm2d(c2))
                if idx < len(nchannels[:-1]) - 1:
                    dec.append(nn.ReLU(True))
        else:
            print("ERROR: unknown mode: {}".format(cfg['mode']))
            sys.exit()

        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def decode(self, Z):
        return self.decoder(Z)

    def encode(self, X):
        return self.encoder(X)

    def forward(self, X):
        Z = self.encoder(X)
        Y = self.decoder(Z)
        return Y, Z

    def get_info(self, key=None):
        if key is None:
            return self.info
        else:
            assert isinstance(key, str), "ERROR in Generator.get_info(): <key> parameters must be an instance of <str>"
            assert key in self.info, "ERROR in Generator.get_info(): unknown <key> parameter"
            return self.info[key]

    def update_info(self, key, value):
        assert isinstance(key, str), "ERROR in Generator.update_info(): <key> parameters must be an instance of <str>"
        # assert key in self.info, "ERROR in Generator.update_info(): unknown <key> parameter"
        self.info[key] = value

    def print_info(self, verbose=False):
        print("\n\n**{}**:\n".format(self.info['model_name']))
        for key in self.info:
            if isinstance(self.info[key], dict):
                print("**{}**\n".format(key))
                for k in self.info[key]:
                    print("{}:\t{}".format(k, self.info[key][k]))
            else:
                if not key == 'model_name':
                    print("**{}**\t{}".format(key, self.info[key]))
        if verbose:
            print("\n\n")
            print(self)
            print("\n\n")

    def training_summary(self):
        print("\n\n**{}**:\n".format(self.info['model_name']))
        for key in self.info:
            if key not in ('model_name', 'model_configuration', 'model_root', 'results_root'):
                if isinstance(self.info[key], dict):
                    print("**{}**\n".format(key))
                    for k in self.info[key]:
                        print("{}:\t{}".format(k, self.info[key][k]))
                else:
                    print("**{}**\t{}".format(key, self.info[key]))
