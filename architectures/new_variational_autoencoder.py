import torch
import torch.nn as nn
import numpy as np
import sys
from torch.nn.modules.upsampling import UpsamplingNearest2d

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
        mode = cfg['mode']
        self.info = info

        #BUILD MODEL
        self.encoder = nn.Sequential([nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(1, 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                      nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(8, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                      nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                      nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                      nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                      nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      View((-1,4096)),
                                      nn.Linear(in_features=4096, out_features=4096, bias=True),
                                      nn.ReLU(True),
                                      nn.Linear(in_features=4096, out_features=512, bias=True),
                                      nn.ReLU(True)])

        self.decoder = nn.Sequential([nn.Linear(in_features=7, out_features=512, bias=True),
                                      nn.ReLU(True),
                                      nn.Linear(in_features=512, out_features=4096, bias=True),
                                      nn.ReLU(True),
                                      nn.Linear(in_features=4096, out_features=4096, bias=True),
                                      nn.ReLU(True),
                                      View((-1,256,4,4)),
                                      UpsamplingNearest2d(size=(8, 8), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(256,128,kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                      nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(8, 8), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(8, 8), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(16, 16), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(8, 8), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(8, 8), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(32, 32), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                      nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(32, 32), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(32, 32), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(64, 64), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                      nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(64, 64), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(64, 64), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(128, 128), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(16, 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                      nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(128, 128), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(128, 128), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(256, 256), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(8, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(8, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                      nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(256, 256), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(True),
                                      UpsamplingNearest2d(size=(256, 256), mode='nearest') if mode == 'resize_conv' else None,
                                      nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) if mode == 'resize_conv'
                                      else nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)])

        self.fcZ_mean = nn.Linear(512,7)
        self.fcZ_logvar = nn.Linear(512,7)


    def encode(self, X):
        X = self.encoder(X)
        return self.fcZ_mean(X), self.fcZ_logvar(X)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, Z):
        return self.decoder(Z)

    def forward(self, X):
        mu, logvar = self.encode(X)
        Z = self.reparameterize(mu, logvar)
        if self.ends_with_sig:
            Y = torch.sigmoid(self.decoder(Z))
        else:
            Y = self.decoder(Z)
        return Y, Z, mu, logvar

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
