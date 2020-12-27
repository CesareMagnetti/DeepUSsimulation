import torch.nn as nn
import numpy as np
import sys
from torch.nn.modules.upsampling import UpsamplingNearest2d as UP


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

########GET INFORMATION ABOUT THE MODEL####################################################################################

        info = {}
        assert 'mode' in cfg, "ERROR: missing mandatory key in cfg: (key: <mode>)"
        assert 'net_name' in cfg, "ERROR: missing mandatory key in cfg: (key: <net_name>)"
        info['model_name'] = cfg['mode'] + "__" + cfg['net_name']
        info['model_configuration'] = cfg
        if 'loss_weights' in cfg:
            k = {}
            assert isinstance(cfg['loss_weights'],dict), "ERROR: 'loss_weights' entry should be a dictionary specifing each weight"
            if len(cfg['loss_weights']) == 2:
                assert ('k1' in cfg['loss_weights'] and 'k2' in cfg['loss_weights']), "ERROR: name of weights should be 'k1' and 'k2'"
                k['k1'] = cfg['loss_weights']['k1']
                k['k2'] = cfg['loss_weights']['k1']
                k['k3'] = cfg['loss_weights']['k2']
            elif len(cfg['loss_weights']) == 3:
                assert ('k1' in cfg['loss_weights'] and 'k2' in cfg['loss_weights'] and 'k3' in cfg['loss_weights'] ),\
                    "ERROR: name of weights should be 'k1' and 'k2' and 'k3'"
                k['k1'] = cfg['loss_weights']['k1']
                k['k2'] = cfg['loss_weights']['k2']
                k['k3'] = cfg['loss_weights']['k3']
            else:
                k['k1'] = 1.
                k['k2'] = 0.
                k['k3'] = 0.1
            self.k = k

        if 'loss_criterions' in cfg:
            criterion = {}
            assert isinstance(cfg['loss_criterions'],dict), "ERROR: 'loss_criterions' entry should be a dictionary specifing each criterion"
            if len(cfg['loss_criterions']) == 2:
                assert ('criterion1' in cfg['loss_criterions'] and 'criterion2' in cfg['loss_criterions']), \
                    "ERROR: name of criterions should be 'criterion1' and 'criterion2'"
                criterion['criterion1'] = cfg['loss_criterions']['criterion1']
                criterion['criterion2'] = cfg['loss_criterions']['criterion1']
                criterion['criterion3'] = cfg['loss_criterions']['criterion2']
            elif len(cfg['loss_criterions']) == 3:
                assert ('criterion1' in cfg['loss_criterions'] and 'criterion2' in cfg['loss_criterions'] and 'criterion3' in cfg['loss_criterions'] ),\
                    "ERROR: name of criterions should be 'criterion1' and 'criterion2' and 'criterion3'"
                criterion['criterion1'] = cfg['loss_criterions']['criterion1']
                criterion['criterion2'] = cfg['loss_criterions']['criterion2']
                criterion['criterion3'] = cfg['loss_criterions']['criterion3']
            else:
                criterion['criterion1'] = nn.MSELoss()
                criterion['criterion2'] = nn.MSELoss()
                criterion['criterion3'] = nn.MSELoss()
            self.criterion = criterion

        info['loss_weights'] = self.k
        info['loss_criterions'] = self.criterion
        self.info = info


#########GET THE PARAMETERS TO BUILD THE NETWORK FROM CFG###############################################################

        nchannels, kernels, strides = cfg['nchannels'], cfg['kernel_size'], cfg['stride']
        padding = tuple(np.int((k - 1) / 2) for k in kernels)
        z_size, nlinear = cfg['z_size'], cfg['nlinear']

        assert nlinear[0] == z_size, "ERROR: first element of nlinear must be equal to z_size"

        unflatten = (nchannels[0],) + cfg['image_size']
        flatten = np.prod(unflatten)

        assert nlinear[-1] == nchannels[0] * cfg['image_size'][0] * cfg['image_size'][1],\
            "ERROR: wrong number of channels/image_size combination"

        if 'batchnorm2d' in cfg:
            batchnorm2d = cfg['batchnorm2d']
        else:
            batchnorm2d = False
        if 'dropout' in cfg:
            dropout = cfg['dropout']
            conv_drop_prob = 0.
            linear_drop_prob = 0.2
        else:
            dropout = False

#########BUILD THE NETWORK##############################################################################################

        dec = []
        enc = []
        # encoding conv layers
        for idx, (c1, c2, stride, ks, pad) in enumerate(
                zip(nchannels[::-1][:-1], nchannels[::-1][1:], strides[::-1], kernels[::-1],padding[::-1])):
            enc.append(nn.Conv2d(c1, c2, stride=stride, kernel_size=ks, padding=pad))
            if dropout:
                enc.append(nn.Dropout2d(conv_drop_prob))
            if batchnorm2d:
                enc.append(nn.BatchNorm2d(c2))
            if idx < len(nchannels[::-1][:-1]) - 1:
                enc.append(nn.ReLU(True))
        # view
        enc.append(View((-1, flatten)))
        # encoding linear layers
        for idx, (c1, c2) in enumerate(zip(nlinear[::-1][:-1], nlinear[::-1][1:])):
            enc.append(nn.Linear(c1, c2))
            if dropout and idx > 0:
                enc.append(nn.Dropout(linear_drop_prob))
            if idx < len(nlinear[:-1]) - 1:
                enc.append(nn.ReLU(True))

        # decoding linear layers
        for idx, (c1, c2) in enumerate(zip(nlinear[:-1], nlinear[1:])):
            dec.append(nn.Linear(c1, c2))
            if dropout and idx > 0:
                dec.append(nn.Dropout(linear_drop_prob))
            if idx < len(nlinear[:-1]) - 1:
                dec.append(nn.ReLU(True))

        # view
        dec.append(View((-1,) + unflatten))
        if cfg['mode'] == "deconv":
            # conv layers
            for idx, (c1, c2, stride, ks, pad) in enumerate(zip(nchannels[:-1], nchannels[1:], strides, kernels, padding)):
                dec.append(nn.ConvTranspose2d(c1, c2, stride=stride, kernel_size=ks, padding=pad))
                if dropout:
                    dec.append(nn.Dropout2d(conv_drop_prob))
                if batchnorm2d:
                    dec.append(nn.BatchNorm2d(c2))
                if idx < len(nchannels[:-1]) - 1:
                    dec.append(nn.ReLU(True))
        elif cfg['mode'] == "resize_conv":
            assert "sizes" in cfg, "ERROR: resample sizes not specified in the configuration"
            sizes = cfg['sizes']
            #FIXME: right now the upsampling convolutions will always be ks=3, stride = 1, padding = 1
            for idx, (c1, c2, size, ks, pad) in enumerate(
                    zip(nchannels[:-1], nchannels[1:], sizes, kernels, padding)):
                dec.append(UP(size))
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

########################################################################################################################

        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

########################################################################################################################
    def decode(self, Z):
        return self.decoder(Z)

    def encode(self, X):
        return self.encoder(X)

    def forward(self, X):
        Z = self.encoder(X)
        Y = self.decoder(Z)
        return Y, Z

    def loss(self, img, target, out1, Z, out2 = None):

        MSE_img_from_img = self.k['k1']*self.criterion['criterion1'](out1, img)
        MSE_Z = self.k['k3']*self.criterion['criterion3'](Z, target)
        if out2 is not None:
            MSE_img_from_z = self.k['k2']*self.criterion['criterion2'](out2, img)
            return MSE_img_from_img, MSE_img_from_z, MSE_Z
        else:
            return MSE_img_from_img, MSE_Z

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
            if key not in ('model_name', 'model_configuration', 'model_root', 'results_root', 'loss_weights', 'loss_criterions'):
                if isinstance(self.info[key], dict):
                    print("**{}**\n".format(key))
                    for k in self.info[key]:
                        print("{}:\t{}".format(k, self.info[key][k]))
                else:
                    print("**{}**\t{}".format(key, self.info[key]))
