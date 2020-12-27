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
        # model attributes
        info = {}
        assert 'mode' in cfg, "ERROR: missing mandatory key in cfg: (key: <mode>)"
        assert 'net_name' in cfg, "ERROR: missing mandatory key in cfg: (key: <net_name>)"
        info['model_name'] = cfg['mode'] + "__" + cfg['net_name']
        info['model_configuration'] = cfg
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
            self.conv_drop_prob = 0.
            self.linear_drop_prob = 0.2
        else:
            self.dropout = False

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
        for idx, (c1, c2) in enumerate(zip(self.nlinear[::-1][:-1], self.nlinear[::-1][1:])):
            enc.append(nn.Linear(c1, c2))
            if self.dropout and idx > 0:
                enc.append(nn.Dropout(self.linear_drop_prob))
            if idx < len(self.nlinear[:-1]) - 1:
                enc.append(nn.ReLU(True))

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
            #FIXME: right now the upsampling convolutions will always be ks=3, stride = 1, padding = 1
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

    def decode(self, Z):
        return self.decoder(Z)

    def encode(self, X):
        return self.encoder(X)

    def forward(self, X):
        Z = self.encoder(X)
        Y = self.decoder(Z)
        return Y, Z

    def freeze_weights(self, mode, layers = None):
        '''
        function to freeze specific layers of the model
        :param mode: <string> either "encoder" or "decoder"
        :param layers: <list/range/tuple> containg layers (id number) to freeze. (default = None, will freeze all layers)

        if unsure of what layer id to pass as an input call function print_info(verbose = True) to inspect model parameters
        '''
        if layers is not None:
            assert isinstance(layers, (range, list,tuple)), \
                "ERROR in freeze_weights(): key (layers) must be an instance of: (range,list,tuple)"

            # freeze specified layers of encorder or decoder
            for name, param in self.named_parameters():
                if mode in name and int(name.split('.')[1]) in layers:
                    param.requires_grad = False
        else:
            # freeze all weights of encorder or decoder
            for name, param in self.named_parameters():
                if mode in name:
                    param.requires_grad = False

    def unfreeze_weights(self, mode = "all", layers=None):
        '''
        function to unfreeze specific layers of the model
        :param mode: <string> either "encoder","decoder" or "all". (default  = "all")
        :param layers: <list/range/tuple> containg layers (id number) to freeze. (default = None, will unfreeze all layers)

        if unsure of what layer id to pass as an input call function print_info(verbose = True) to inspect model parameters
        '''
        if layers is not None:
            assert isinstance(layers, (range, list, tuple)), \
                "ERROR in freeze_weights(): key (layers) must be an instance of: (range,list,tuple)"

            # unfreeze specified layers of encorder or decoder
            for name, param in self.named_parameters():
                if mode in name and int(name.split('.')[1]) in layers:
                    param.requires_grad = True
        else:
            # unfreeze all weights of encorder or decoder
            for name, param in self.named_parameters():
                if mode in name or mode == "all":
                    param.requires_grad = True

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
        '''
        prints a summary of the training epoch (prints all relevant info that changes through epochs):

        expected output:

        **EPOCH RESULTS**


        **model_name**  deconv__AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1

        **epoch**	23
        **best_val_loss**	0.3724544048309326
        **best_train_loss**	0.052513159811496735
        **val_loss**	0.3817540109157562
        **train_loss**	0.052513159811496735
        '''

        print("\n**EPOCH RESULTS**")
        print("\n\n**{}**\t{}:\n".format('model_name',self.info['model_name']))
        for key in self.info:
            if key not in ('model_name', 'model_configuration', 'model_root', 'results_root', 'old_info'):
                if isinstance(self.info[key], dict):
                    print("**{}**\n".format(key))
                    for k in self.info[key]:
                        print("{}:\t{}".format(k, self.info[key][k]))
                else:
                    print("**{}**\t{}".format(key, self.info[key]))

# class Generator(nn.Module):
#     def __init__(self, cfg):
#         super(Generator, self).__init__()
#         # model attributes
#         info = {}
#         assert 'mode' in cfg, "ERROR: missing mandatory key in cfg: (key: <mode>)"
#         assert 'net_name' in cfg, "ERROR: missing mandatory key in cfg: (key: <net_name>)"
#         info['model_name'] = cfg['mode'] + "__" + cfg['net_name']
#         info['model_configuration'] = cfg
#         self.info = info
#         self.nchannels = cfg['nchannels']
#         self.ks = cfg['kernel_size']
#         self.strides = cfg['stride']
#         self.padding = tuple(np.int((k - 1) / 2) for k in self.ks)
#         self.z_size = cfg['z_size']
#         self.nlinear = cfg['nlinear']
#         assert self.nlinear[0] == self.z_size, "ERROR: first element of nlinear must be equal to z_size"
#         self.unflatten = (self.nchannels[0],) + cfg['image_size']
#         self.flatten = np.prod(self.unflatten)
#         assert self.nlinear[-1] == self.nchannels[0] * cfg['image_size'][0] * cfg['image_size'][
#             1], "ERROR: wrong number of channels/image_size combination"
#         if 'batchnorm2d' in cfg:
#             self.batchnorm2d = cfg['batchnorm2d']
#         else:
#             self.batchnorm2d = False
#         if 'dropout' in cfg:
#             self.dropout = cfg['dropout']
#         else:
#             self.dropout = False
#
#         dec = []
#         enc = []
#         # encoding conv layers
#         for idx, (c1, c2, stride, ks, padding) in enumerate(
#                 zip(self.nchannels[::-1][:-1], self.nchannels[::-1][1:], self.strides[::-1], self.ks[::-1],
#                     self.padding[::-1])):
#             enc.append(nn.Conv2d(c1, c2, stride=stride, kernel_size=ks, padding=padding))
#
#             if idx < len(self.nchannels[::-1][:-1]) - 1:
#                 if self.batchnorm2d:
#                     enc.append(nn.BatchNorm2d(c2))
#                 enc.append(nn.ReLU(True))
#                 if self.dropout:
#                     enc.append(nn.Dropout2d(0.1))
#         # view
#         enc.append(View((-1, self.flatten)))
#         # encoding linear layers
#         for idx, (c1, c2) in enumerate(zip(self.nlinear[::-1][:-1], self.nlinear[::-1][1:])):
#             if self.dropout and idx < len(self.nlinear[:-1]) - 1:
#                 enc.append(nn.Dropout())
#             enc.append(nn.Linear(c1, c2))
#             if idx < len(self.nlinear[:-1]) - 1:
#                 enc.append(nn.ReLU(True))
#
#         # decoding linear layers
#         for idx, (c1, c2) in enumerate(zip(self.nlinear[:-1], self.nlinear[1:])):
#             dec.append(nn.Linear(c1, c2))
#             if idx < len(self.nlinear[:-1]) - 1:
#                 dec.append(nn.ReLU(True))
#
#         # view
#         dec.append(View((-1,) + self.unflatten))
#         if cfg['mode'] == "deconv":
#             # conv layers
#             for idx, (c1, c2, stride, ks, padding) in enumerate(
#                     zip(self.nchannels[:-1], self.nchannels[1:], self.strides, self.ks, self.padding)):
#                 dec.append(nn.ConvTranspose2d(c1, c2, stride=stride, kernel_size=ks, padding=padding))
#                 if idx < len(self.nchannels[:-1]) - 1:
#                     if self.batchnorm2d:
#                         dec.append(nn.BatchNorm2d(c2))
#                     dec.append(nn.ReLU(True))
#         elif cfg['mode'] == "resize_conv":
#             assert "sizes" in cfg, "ERROR: resample sizes not specified in the configuration"
#             self.sizes = cfg['sizes']
#             #FIXME: right now the upsampling convolutions will always be ks=3, stride = 1, padding = 1
#             for idx, (c1, c2, size, kernel, pad) in enumerate(
#                     zip(self.nchannels[:-1], self.nchannels[1:], self.sizes, self.ks, self.padding)):
#                 dec.append(UP(size))
#                 dec.append(nn.Conv2d(c1, c2, stride=1, kernel_size=3, padding=1))
#                 if idx < len(self.nchannels[:-1]) - 1:
#                     if self.batchnorm2d:
#                         dec.append(nn.BatchNorm2d(c2))
#                     dec.append(nn.ReLU(True))
#         else:
#             print("ERROR: unknown mode: {}".format(cfg['mode']))
#             sys.exit()
#
#         ##last layer
#         # dec.append(nn.Conv2d(self.nchannels[-1], 1, stride = 1, kernel_size = 3, padding = 1))
#         self.encoder = nn.Sequential(*enc)
#         self.decoder = nn.Sequential(*dec)
#
#     def decode(self, Z):
#         return self.decoder(Z)
#
#     def encode(self, X):
#         return self.encoder(X)
#
#     def forward(self, X):
#         Z = self.encoder(X)
#         Y = self.decoder(Z)
#         return Y, Z
#
#     def freeze_weights(self, mode, layers = None):
#         '''
#         function to freeze specific layers of the model
#         :param mode: <string> either "encoder" or "decoder"
#         :param layers: <list/range/tuple> containg layers (id number) to freeze. (default = None, will freeze all layers)
#
#         if unsure of what layer id to pass as an input call function print_info(verbose = True) to inspect model parameters
#         '''
#         if layers is not None:
#             assert isinstance(layers, (range, list,tuple)), \
#                 "ERROR in freeze_weights(): key (layers) must be an instance of: (range,list,tuple)"
#
#             # freeze specified layers of encorder or decoder
#             for name, param in self.named_parameters():
#                 if mode in name and int(name.split('.')[1]) in layers:
#                     param.requires_grad = False
#         else:
#             # freeze all weights of encorder or decoder
#             for name, param in self.named_parameters():
#                 if mode in name:
#                     param.requires_grad = False
#
#     def unfreeze_weights(self, mode = "all", layers=None):
#         '''
#         function to unfreeze specific layers of the model
#         :param mode: <string> either "encoder","decoder" or "all". (default  = "all")
#         :param layers: <list/range/tuple> containg layers (id number) to freeze. (default = None, will unfreeze all layers)
#
#         if unsure of what layer id to pass as an input call function print_info(verbose = True) to inspect model parameters
#         '''
#         if layers is not None:
#             assert isinstance(layers, (range, list, tuple)), \
#                 "ERROR in freeze_weights(): key (layers) must be an instance of: (range,list,tuple)"
#
#             # unfreeze specified layers of encorder or decoder
#             for name, param in self.named_parameters():
#                 if mode in name and int(name.split('.')[1]) in layers:
#                     param.requires_grad = True
#         else:
#             # unfreeze all weights of encorder or decoder
#             for name, param in self.named_parameters():
#                 if mode in name or mode == "all":
#                     param.requires_grad = True
#
#     def get_info(self, key=None):
#         if key is None:
#             return self.info
#         else:
#             assert isinstance(key, str), "ERROR in Generator.get_info(): <key> parameters must be an instance of <str>"
#             assert key in self.info, "ERROR in Generator.get_info(): unknown <key> parameter"
#             return self.info[key]
#
#     def update_info(self, key, value):
#         assert isinstance(key, str), "ERROR in Generator.update_info(): <key> parameters must be an instance of <str>"
#         # assert key in self.info, "ERROR in Generator.update_info(): unknown <key> parameter"
#         self.info[key] = value
#
#     def print_info(self, verbose=False):
#         print("\n\n**{}**:\n".format(self.info['model_name']))
#         for key in self.info:
#             if isinstance(self.info[key], dict):
#                 print("**{}**\n".format(key))
#                 for k in self.info[key]:
#                     print("{}:\t{}".format(k, self.info[key][k]))
#             else:
#                 if not key == 'model_name':
#                     print("**{}**\t{}".format(key, self.info[key]))
#         if verbose:
#             print("\n\nMODEL ARCHITECTURE:\n\n")
#             print(self)
#             print("\n\nMODEL PARAMETERS:\n\n")
#             for name, param in self.named_parameters():
#                 print('name: ', name)
#                 print(type(param))
#                 print('param.shape: ', param.shape)
#                 print('param.requires_grad: ', param.requires_grad)
#                 print('=====')
#             print("\n\n")
#
#     def training_summary(self):
#         '''
#         prints a summary of the training epoch (prints all relevant info that changes through epochs):
#
#         expected output:
#
#         **EPOCH RESULTS**
#
#
#         **model_name**  deconv__AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1
#
#         **epoch**	23
#         **best_val_loss**	0.3724544048309326
#         **best_train_loss**	0.052513159811496735
#         **val_loss**	0.3817540109157562
#         **train_loss**	0.052513159811496735
#         '''
#
#         print("\n**EPOCH RESULTS**")
#         print("\n\n**{}**\t{}:\n".format('model_name',self.info['model_name']))
#         for key in self.info:
#             if key not in ('model_name', 'model_configuration', 'model_root', 'results_root', 'old_info'):
#                 if isinstance(self.info[key], dict):
#                     print("**{}**\n".format(key))
#                     for k in self.info[key]:
#                         print("{}:\t{}".format(k, self.info[key][k]))
#                 else:
#                     print("**{}**\t{}".format(key, self.info[key]))
#
