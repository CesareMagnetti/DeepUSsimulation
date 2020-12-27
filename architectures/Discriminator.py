import torch.nn as nn
import torch
import numpy as np

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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is B x 1 x 256 x 256
            nn.Conv2d(1, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. B x 32 x 128 x 128
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. B x 64 x 64 x 64
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. B x 128 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. B x 256 x 16 x 16
            nn.Conv2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. B x 512 x 4 X 4
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            View((-1,100)),
            nn.Linear(100,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Conditional_Discriminator(nn.Module):
    def __init__(self, z_size=7,latent_image_size=(32,8,8)):
        super(Conditional_Discriminator, self).__init__()

        self.fc_tracker = nn.Sequential(nn.Linear(z_size,32),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(32,256),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(256, 512),
                                        nn.LeakyReLU(0.2, inplace=True))

        self.fc = nn.Sequential(nn.Linear(2048 + 512,1024),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(1024,512),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(512,1),
                                nn.Sigmoid())

        self.conv = nn.Sequential(nn.Conv2d(1, 32, stride=2, kernel_size=4, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.LeakyReLU(0.2, inplace=True),

                                  nn.Conv2d(32, 32, stride=2, kernel_size=4, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.LeakyReLU(0.2, inplace=True),

                                  nn.Conv2d(32, 32, stride=2, kernel_size=4, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.LeakyReLU(0.2, inplace=True),

                                  nn.Conv2d(32, 32, stride=2, kernel_size=4, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.LeakyReLU(0.2, inplace=True),

                                  nn.Conv2d(32, 32, stride=2, kernel_size=4, padding=1),
                                  View((-1,np.prod(latent_image_size))))

    def forward(self, X, t):
        y_ = self.fc_tracker(t)
        X = self.conv(X)
        X=self.fc(torch.cat([X, y_], 1))
        return X


class Latent_Discriminator(nn.Module):
    def __init__(self, z_size=7, N=1000):
        super(Latent_Discriminator, self).__init__()
        self.main = nn.Sequential(nn.Linear(z_size,N),
                                  nn.ReLU(True),
                                  nn.Linear(N,N),
                                  nn.ReLU(True),
                                  nn.Linear(N,1),
                                  nn.Sigmoid())

    def forward(self,z):
        return self.main(z)

