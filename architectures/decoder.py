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
        info = {}
        assert 'mode' in cfg, "ERROR: missing mandatory key in cfg: (key: <mode>)"
        assert 'net_name' in cfg, "ERROR: missing mandatory key in cfg: (key: <net_name>)"
        info['model_name'] = cfg['mode'] + "__" + cfg['net_name']
        info['model_configuration'] = cfg
        self.info = info
        self.name = cfg['mode'] + cfg['net_name']
        self.nchannels = cfg['nchannels']
        self.ks = cfg['kernel_size']
        self.strides = cfg['stride']
        self.padding = tuple(np.int((k - 1) / 2) for k in self.ks)
        self.z_size = cfg['z_size']
        self.nlinear = cfg['nlinear']
        assert self.nlinear[0] == self.z_size, "ERROR: first element of nlinear must be equal to z_size"
        self.unflatten = (self.nchannels[0],) + cfg['image_size']
        assert self.nlinear[-1] == self.nchannels[0] * cfg['image_size'][0] * cfg['image_size'][
            1], "ERROR: wrong number of channels/image_size combination"
        if 'batchnorm2d' in cfg:
            self.batchnorm2d = cfg['batchnorm2d']
        else:
            self.batchnorm2d = False
        if 'dropout' in cfg:
            self.dropout = cfg['dropout']
            self.conv_drop_prob = 0.1
            self.linear_drop_prob = 0.5
        else:
            self.dropout = False

        l = []
        # linear layers
        for idx, (c1, c2) in enumerate(zip(self.nlinear[:-1], self.nlinear[1:])):
            l.append(nn.Linear(c1, c2))
            if self.dropout and idx > 0:
                l.append(nn.Dropout(self.linear_drop_prob))
            if idx < len(self.nlinear[:-1]) - 1:
                l.append(nn.ReLU(True))

        # view
        l.append(View((-1,) + self.unflatten))
        if cfg['mode'] == "deconv":
            # conv layers
            for idx, (c1, c2, stride, ks, padding) in enumerate(zip(self.nchannels[:-1], self.nchannels[1:], self.strides, self.ks,
                                                   self.padding)):
                l.append(nn.ConvTranspose2d(c1, c2, stride=stride, kernel_size=ks, padding=padding))
                if idx < len(self.nchannels[:-1]) - 1:
                    if self.dropout:
                        l.append(nn.Dropout2d(self.conv_drop_prob))
                    if self.batchnorm2d:
                        l.append(nn.BatchNorm2d(c2))
                    l.append(nn.ReLU(True))

        elif cfg['mode'] == "resize_conv":
            assert "sizes" in cfg, "ERROR: resample sizes not specified in the configuration"
            # assert cfg['sizes'][0] == cfg['image_size'], "ERROR: first size must be equal to cfg['image_size']"
            self.sizes = cfg['sizes']
            for idx, (c1, c2, size, kernel, pad) in enumerate(zip(self.nchannels[:-1], self.nchannels[1:], self.sizes, self.ks,
                                                 self.padding)):
                l.append(UP(size))
                l.append(nn.Conv2d(c1, c2, stride=1, kernel_size=kernel, padding=pad))

                if idx < len(self.nchannels[:-1]) - 1:
                    if self.dropout:
                        l.append(nn.Dropout2d(self.conv_drop_prob))
                    if self.batchnorm2d:
                        l.append(nn.BatchNorm2d(c2))
                    l.append(nn.ReLU(True))
        else:
            print("ERROR: unknown mode: {}".format(cfg['mode']))
            sys.exit()

        # last layer
        #l.append(nn.Conv2d(self.nchannels[-1], 1, stride=1, kernel_size=3, padding=1))
        if 'end_with_sigmoid' in cfg:
            assert isinstance(cfg['end_with_sigmoid'], bool), "ERROR in cfg: key <end_with_sigmoid> must be a bool (True/False)"
            l.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*l)

    def forward(self, X):
        return self.decoder(X)

    def decode(self, X):
        return self.decoder(X)

    # def load_my_state_dict(self, state_dict):
    #
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name not in own_state:
    #             continue
    #         if isinstance(param, Parameter):
    #             # backwards compatibility for serialized parameters
    #             param = param.data
    #         own_state[name].copy_(param)

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
            print("\n\nMODEL ARCHITECTURE:\n\n")
            print(self)
            print("\n\nMODEL PARAMETERS:\n\n")
            for name, param in self.named_parameters():
                print('name: ', name)
                print(type(param))
                print('param.shape: ', param.shape)
                print('param.requires_grad: ', param.requires_grad)
                print('=====')
            print("\n\n")

    def training_summary(self):
        print("\n\n**{}**:\n".format(self.info['model_name']))
        for key in self.info:
            if key not in ('model_name', 'model_configuration', 'model_root', 'results_root', 'old_info'):
                if isinstance(self.info[key], dict):
                    print("**{}**\n".format(key))
                    for k in self.info[key]:
                        print("{}:\t{}".format(k, self.info[key][k]))
                else:
                    print("**{}**\t{}".format(key, self.info[key]))
