'''
Intuition : 
    In face emotion transfer tasks, we can copy a lot of information from input to output image. 
    By passing the information directly, I hope to make the learning problem simpler and 
    hence requiring signifcantly fewer parameters.  
'''

import functools
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
from torchsummary import summary
import numpy as np

class MaskNet(nn.Module):
    '''
    Model maps (w, h, ch) to (w, h, ch) with a conv net + a sigmoid. Computes a simple image attention matrix. 
    '''
    def __init__(self, channels = 3):
        super(MaskNet, self).__init__()
        self.model = nn.Sequential(*[
                        nn.Conv2d(channels, 64, 5, 1, 2),
                        nn.BatchNorm2d(64), 
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(64, channels, 5, 1, 2), 
                        nn.Sigmoid()                
        ])
    def forward(self, x):
        return self.model(x)


class SynNet(nn.Module):
    '''
    Model maps (w, h, ch) to (w, h, ch) with a conv net. Responsible for imagining new stuff.
    '''
    def __init__(self, channels):
        super(SynNet, self).__init__()
        self.model = nn.Sequential(*[
                        nn.Conv2d(channels, 64, 5, 1, 2),
                        nn.BatchNorm2d(64), 
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(64, channels, 5, 1, 2), 
                        nn.Tanh()                
        ])
    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    '''
    Maps (w, h, ch) to (w, h, ch) using mask and syn
    mask = mask_net(input)
    predictions = mask * inputs + (1-mask) * syn
    '''
    def __init__(self, in_channels):
        super(Generator, self).__init__()
        self.mask_net, self.syn_net = MaskNet(in_channels), SynNet(in_channels)

    def forward(self, x):
        mask = self.mask_net(x)
        synthesized = self.syn_net(x)
        copied = mask * x
        weighted_synthesized = (1 - mask) * synthesized
        return copied + weighted_synthesized, mask, copied, synthesized, weighted_synthesized 

class Discriminator(nn.Module):
    '''
    Maps (w, h, 3) -> flattened sequence of sigmoids. 
    '''
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 4, 2, 1),
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



if __name__ == '__main__':
    # Test mask net
    print(MaskNet(3).forward(torch.randn((5, 3, 28, 28))).shape)
    print(MaskNet(1).forward(torch.randn((5, 1, 64, 64))).shape)

    # Test SynNet
    print(SynNet(3).forward(torch.randn((5, 3, 28, 28))).shape)
    print(SynNet(1).forward(torch.randn((5, 1, 64, 64))).shape)

    # Test Generator
    
    generator = Generator(1)
    print(generator(torch.randn((5, 1, 28, 28)))[0].shape)

    # Test Discriminator
    print(Discriminator(1)(generator(torch.randn((5, 1, 28, 28)))[0]).shape)

