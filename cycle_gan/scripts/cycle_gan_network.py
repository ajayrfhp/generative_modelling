import functools
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
from torchsummary import summary
import numpy as np


class Generator(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, num_blocks = 3):
        super(Generator, self).__init__()
        self.input_convs = nn.Sequential(*[ 
                            nn.Conv2d(in_channels, 64, 7, 1, 3), 
                            nn.Conv2d(64, 64, 3, 2, 1), 
                            nn.Conv2d(64, 128, 3, 2, 1)
                            ])
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(num_blocks)])
        self.output_convs = nn.Sequential(*[
                            nn.ConvTranspose2d(128, 64, 3, 2, padding = 1, output_padding = 1),
                            nn.ConvTranspose2d(64, 64, 3, 2, padding = 1, output_padding = 1),
                            nn.Conv2d(64, out_channels, 7, 1, padding = 3),
                            nn.Tanh()   
        ])
        self.model = nn.Sequential(self.input_convs, self.res_blocks, self.output_convs)
        
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid(),
        ])
    
    def forward(self, x):
        x = self.model(x)
        return x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.conv1 = nn.Conv2d(filters, filters, 3, 1, 1)
        self.conv2 = nn.Conv2d(filters, filters, 3, 1, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = nn.BatchNorm2d(self.filters)(x1)
        x1 = nn.LeakyReLU(0.2)(x1)
        x2 = self.conv2(x1)
        x2 = nn.BatchNorm2d(self.filters)(x2)
        return nn.LeakyReLU(0.2)(x2 + x)

        

if __name__ == '__main__':
    # Test Forward pass for CelebA dataset
    G = Generator(1, 1, num_blocks = 3)
    D = Discriminator(1)
    print(D(G(torch.randn(5, 1, 28, 28))).shape)
