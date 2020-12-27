import torch.nn as nn
import torch
from torch.nn.modules.upsampling import UpsamplingNearest2d as UP
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
tensortype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

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


class Conditional_Decoder(nn.Module):
    '''
    Conditional decoder adversarial generator based on the best performing deterministic decoder found through the thesis.
    ARCHITECTURE  OF THE DECODER:

    linear layers: 7-31-256-512-1024-2048, all except the last one coupled with ReLU.

    reshape from Bx2048 to Bx32x8x8

    convolutional layers:    32-----32-----32-------32-------32----1, all layers are preceded by 2D Nearest Neighbour
    with sizes:         16x16--32x32--64x64--128x128--256x256, and followed by BatchNorm2D and ReLU (except last layer)
    batch norm (BN):           BN(32)-BN(32)-BN(32)---BN(32)
    ReLU:                      ReLU---ReLU---ReLU------ReLU

    The GAN approach combines random noise (n) at the 4th linear layer level:
    expand 7D tracker as 7-32-256-512
    continue linear layers as 512+noise-1024-2048.
    convolutional layers are the same.

    DCGAN paper suggests the use of a tanh function to map output to range (-1,1), experiments about this will be conducted.
    '''
    def __init__(self, z_size=7, noise_size=100, latent_image_size = (32,8,8)):
        super(Conditional_Decoder, self).__init__()
        info = {}
        info['model_configuration']={}
        info['model_configuration']['z_size'] = z_size
        info['model_configuration']['noise_size'] = noise_size
        info['model_configuration']['latent_image_size'] = latent_image_size
        info['model_configuration']['mode'] = "resize_conv"
        info['model_name'] = "CONDITIONAL_DECODER__LINEAR_{:.0f}_32_256_512__512+{:.0f}_1024_2048__CONV_32x5_1".format(z_size,noise_size)
        self.info = info

        self.fc_tracker = nn.Sequential(nn.Linear(z_size,32),
                                        nn.ReLU(True),
                                        nn.Linear(32,256),
                                        nn.ReLU(True),
                                        nn.Linear(256, 512),
                                        nn.ReLU(True))

        self.fc = nn.Sequential(nn.Linear(noise_size+512,1024),
                                nn.ReLU(True),
                                nn.Linear(1024,2048))

        self.conv = nn.Sequential(View((-1,) + latent_image_size),
                                  UP((16,16)),
                                  nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(True),
                                  UP((32, 32)),
                                  nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(True),
                                  UP((64, 64)),
                                  nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(True),
                                  UP((128, 128)),
                                  nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(True),
                                  UP((256, 256)),
                                  nn.Conv2d(32, 1, stride=1, kernel_size=3, padding=1))

    def forward(self, n, t):
        y_ = self.fc_tracker(t)
        X = self.fc(torch.cat([n, y_], 1))
        X = self.conv(X)
        return X

    def decode(self, t, n = None):
        if n is None:
            n= torch.randn(1, 100, device=device).type(tensortype)

        y_ = self.fc_tracker(t)
        X = self.fc(torch.cat([n, y_], 1))
        X = self.conv(X)
        return X

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


class Conditional_Adversarial_Autoencoder(nn.Module):
    '''
    Conditional Autoencoder adversarial generator based on the best performing deterministic decoder found through the thesis.

    ARCHITECTURE OF THE ENCODER:

        -conv layers: 1-32-32-32-32-32. all with kernel size = 4 stride = 2 and padding = 1. all coupled (except last one)
         with batchnorm 2D and ReLU
        -rehsape from Bx32x8x8 to Bx2048
        -linear layers: 2048-1024-512-256-7+Noise_size all (except last one) coupled with ReLU.

    ARCHITECTURE  OF THE DECODER:

        -expand 7D tracker as 7-32-256-512
        -continue linear layers as 512+noise-1024-2048.
        -reshape from Bx2048 to Bx32x8x8
        -convolutional layers: 32-32-32-32-32-1. all with kernel size = 3, stride&padding = 1. all layers preceded by
         2D-Upsampling Nearest neighbour to double the image size and followed by batchnorm2d+ReLU (except last one)

    DCGAN paper suggests the use of a tanh function to map output to range (-1,1), or sigmoid to map to (0,1)
    experiments about this will be conducted.
    '''
    def __init__(self, z_size=7, noise_size=100, latent_image_size = (32,8,8)):
        super(Conditional_Adversarial_Autoencoder, self).__init__()
        info = {}
        info['model_configuration']={}
        info['model_configuration']['z_size'] = z_size
        info['model_configuration']['noise_size'] = noise_size
        info['model_configuration']['latent_image_size'] = latent_image_size
        info['model_configuration']['mode'] = "resize_conv"
        info['model_name'] = "CONDITIONAL_ADVERSARIAL_AUTOENCODER__LINEAR_{:.0f}_32_256_512__512+{:.0f}_1024_2048__CONV_32x5_1".format(z_size,noise_size)
        self.info = info

        self.conv_encoder = nn.Sequential(nn.Conv2d(1,32, stride=2,kernel_size=4, padding=1), #size 256x256
                                          nn.BatchNorm2d(32),
                                          nn.ReLU(True), #size 128x128
                                          nn.Conv2d(32, 32, stride=2, kernel_size=4, padding=1),
                                          nn.BatchNorm2d(32),
                                          nn.ReLU(True),#size 64x64
                                          nn.Conv2d(32, 32, stride=2, kernel_size=4, padding=1),
                                          nn.BatchNorm2d(32),
                                          nn.ReLU(True), #size 32x32
                                          nn.Conv2d(32, 32, stride=2, kernel_size=4, padding=1),
                                          nn.BatchNorm2d(32),
                                          nn.ReLU(True), #size 16x16
                                          nn.Conv2d(32, 32, stride=2, kernel_size=4, padding=1) #size 8x8
                                          )

        self.fc_encoder = nn.Sequential(View((-1, np.prod(latent_image_size))),
                                        nn.Linear(np.prod(latent_image_size),1024),
                                        nn.ReLU(True),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512,256),
                                        nn.ReLU(True),
                                        nn.Linear(256, z_size+noise_size)
                                        )

        self.fc_decoder_tracker = nn.Sequential(nn.Linear(z_size,32),
                                        nn.ReLU(True),
                                        nn.Linear(32,256),
                                        nn.ReLU(True),
                                        nn.Linear(256, 512),
                                        nn.ReLU(True))

        self.fc_decoder = nn.Sequential(nn.Linear(noise_size+512,1024),
                                nn.ReLU(True),
                                nn.Linear(1024,2048))

        self.conv_decoder = nn.Sequential(View((-1,) + latent_image_size),
                                  UP((16,16)),
                                  nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(True),
                                  UP((32, 32)),
                                  nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(True),
                                  UP((64, 64)),
                                  nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(True),
                                  UP((128, 128)),
                                  nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(True),
                                  UP((256, 256)),
                                  nn.Conv2d(32, 1, stride=1, kernel_size=3, padding=1))

    def forward(self, X):

        #encode input image
        z = self.fc_encoder(self.conv_encoder(X)) #z[:z_size] is tracker, x[z_size:] is latent noise

        #decode tracker only
        y_ = self.fc_decoder_tracker(z[:,:self.info['model_configuration']['z_size']])

        #generate image from noise + expanded tracker
        X = self.conv_decoder(self.fc_decoder(torch.cat([z[:,self.info['model_configuration']['z_size']:], y_], 1)))

        #return recunstructed image, target z and noise z
        return X, z[:,:self.info['model_configuration']['z_size']], z[:,self.info['model_configuration']['z_size']:]

    def decode(self, t, n = None):

        #generate random noise if not given (inference time)
        if n is None:
            n= torch.randn(1, 100, device=device).type(tensortype)

        #decode tracker only
        y_ = self.fc_decoder_tracker(t)

        # generate image from noise + expanded tracker
        X = self.conv_decoder(self.fc_decoder(torch.cat([n, y_], 1)))

        return X


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
