import torch
import torch.nn as nn
import numpy as np
import sys
from torch.nn.modules.upsampling import UpsamplingNearest2d as UP
import torch.nn.init as init

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

def reparametrize(mu, logvar, training=True):
    r"""Reparametrization trick used in variational inference for autoencoders

    This is used for the parametrisation of the latent variable vector so that it becomes differentiable

    https://arxiv.org/pdf/1312.6114.pdf


    .. math::
        x = \mu + \epsilon, \quad \epsilon \sim N(0,logvar)


    Args:
        mu (tensor) Input expectation values
        logvar (tensor) Input log-variance values
        training (bool) If true the returned value is the expectation augmented with random variable of log-variance
          logvar, if false the expectation alone is returned
    """
    if training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    else:
        return mu


def kaiming_init(m):
    r"""Fills the input `Tensor` with values according to the method
    described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \sqrt{\frac{6}{(1 + a^2) \times \text{fan\_in}}}

    Also known as He initialization.

    Examples:
        >>> module = nn.Conv2D(16,32)
        >>> kaiming_init(module)

    Args:
        m (nn.Module) module where weights need initialisation
    """

    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class Generator(nn.Module):
    """Modified architecture inspired from beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018).

       credits to Alberto Gomez, KCL

       This models takes an input configuration (cfg) and builds the correspondin VAE.

       Example of cfg:

       ALL HYPERPARAMETERS ARE GIVEN WITH REFERENCE TO THE DECODER (FROM HIDDEN LAYERS TO OUTPUT)

        cfg = {"nchannels": (32, 32, 32, 32, 32, 32, 1), #channels of the convolutional part of the VAE
               "kernel_size": (4, 4, 4, 4, 4, 1), #Kernel sizes of each conv layer
                                                  #(for "deconv" they will be mirrored in the decoder)
                                                 #(for "deconv" they will be mirrored in the decoder)
               "stride": (2, 2, 2, 2, 2, 1), #Strides of each conv layer
               "batchnorm2d": True, #if true will add nn.BatchNorm2d() after each conv layer
               "z_size": 7,  #latent space size
               "image_size": (8,8) #latent image size
               "nlinear": (7, 32, 512, 1024, 2048), #features of the fully connected layer
               'end_with_sigmoid': True, #if true appends nn.Sigmoid() as a last layer after the decoder
               "BETA": 0.00001, #higher number gives higher kldiv contribution
               "net_name": "VAE__LINEAR_7_32_512_1024_2048__CONV_32x6_1", #optional: name of the network
               }

        .. todo :: Check the generalisation of the papameterization
        .. todo :: Extend documentation, Adjust Resize_Conv mode to receive input kernels,strides etc

    """


    def __init__(self, cfg):
        super(Generator, self).__init__()
        # model attributes
        info = {}
        assert 'mode' in cfg, "ERROR: missing mandatory key in cfg: (key: <mode>)"
        assert 'net_name' in cfg, "ERROR: missing mandatory key in cfg: (key: <net_name>)"
        info['model_name'] = cfg['mode'] + "__" + cfg['net_name']
        info['model_configuration'] = cfg
        info["BETA"] = cfg['BETA']
        self.info = info
        self.nchannels = cfg['nchannels']
        self.ks = cfg['kernel_size']
        self.strides = cfg['stride']
        self.padding = tuple(np.int((k - 1) / 2) for k in self.ks)
        self.z_size = cfg['z_size']
        self.nlinear = cfg['nlinear']
        assert self.nlinear[0] == self.z_size, "ERROR: first element of nlinear must be equal to z_size"
        self.unflatten = (self.nchannels[0],) + cfg['image_size']
        self.flatten = np.prod(self.unflatten)
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
        if 'ends_with_sig' in cfg:
            self.ends_with_sig = cfg['ends_with_sig']
        else:
            self.ends_with_sig  = False

        dec = []
        enc = []
        # encoding conv layers
        for idx, (c1, c2, stride, ks, padding) in enumerate(
                zip(self.nchannels[::-1][:-1], self.nchannels[::-1][1:], self.strides[::-1], self.ks[::-1],
                    self.padding[::-1])):
            enc.append(nn.Conv2d(c1, c2, stride=stride, kernel_size=ks, padding=padding))
            if self.dropout:
                enc.append(nn.Dropout2d(self.conv_drop_prob))
            if self.batchnorm2d:
                enc.append(nn.BatchNorm2d(c2))
            if idx < len(self.nchannels[::-1][:-1]) - 1:
                enc.append(nn.ReLU(True))
        # view
        enc.append(View((-1, self.flatten)))
        # encoding linear layers
        for idx, (c1, c2) in enumerate(zip(self.nlinear[::-1][:-2], self.nlinear[::-1][1:-1])):
            enc.append(nn.Linear(c1, c2))
            if self.dropout and idx > 0:
                enc.append(nn.Dropout(self.linear_drop_prob))
            enc.append(nn.ReLU(True))

        #append last encoding layer
        enc.append(nn.Linear(self.nlinear[1], 2*self.z_size))


        # decoding linear layers
        for idx, (c1, c2) in enumerate(zip(self.nlinear[:-1], self.nlinear[1:])):
            dec.append(nn.Linear(c1, c2))
            if self.dropout and idx > 0:
                dec.append(nn.Dropout(self.linear_drop_prob))
            if idx < len(self.nlinear[:-1]) - 1:
                dec.append(nn.ReLU(True))

        # view
        dec.append(View((-1,) + self.unflatten))
        if cfg['mode'] == "deconv":
            # conv layers
            for idx, (c1, c2, stride, ks, padding) in enumerate(
                    zip(self.nchannels[:-1], self.nchannels[1:], self.strides, self.ks, self.padding)):
                dec.append(nn.ConvTranspose2d(c1, c2, stride=stride, kernel_size=ks, padding=padding))
                if idx < len(self.nchannels[:-1]) - 1:
                    if self.dropout:
                        dec.append(nn.Dropout2d(self.conv_drop_prob))
                    if self.batchnorm2d:
                        dec.append(nn.BatchNorm2d(c2))
                    dec.append(nn.ReLU(True))
        elif cfg['mode'] == "resize_conv":
            assert "sizes" in cfg, "ERROR: resample sizes not specified in the configuration"
            self.sizes = cfg['sizes']
            ##FIXME: right now the upsampling convolutions will always be ks=3, stride = 1, padding = 1
            for idx, (c1, c2, size, kernel, pad) in enumerate(
                    zip(self.nchannels[:-1], self.nchannels[1:], self.sizes, self.ks, self.padding)):
                dec.append(UP(size))
                dec.append(nn.Conv2d(c1, c2, stride=1, kernel_size=3, padding=1))
                if idx < len(self.nchannels[:-1]) - 1:
                    if self.dropout:
                        dec.append(nn.Dropout2d(self.conv_drop_prob))
                    if self.batchnorm2d:
                        dec.append(nn.BatchNorm2d(c2))
                    dec.append(nn.ReLU(True))
        else:
            print("ERROR: unknown mode: {}".format(cfg['mode']))
            sys.exit()

        ##last layer
        # dec.append(nn.Conv2d(self.nchannels[-1], 1, stride = 1, kernel_size = 3, padding = 1))
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)



    #member functions
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def encode(self, X):
        X = self.encoder(X)
        return X[:,:self.z_size], X[:,self.z_size:]


    def decode(self, Z):
        return self.decoder(Z)

    def forward(self, X):
        mu, logvar = self.encode(X)
        Z = reparametrize(mu, logvar, self.training)
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
