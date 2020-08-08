
import functools
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
from torchsummary import summary
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

sys.path.append('../')
from utils.autoregressive_utils import visualize_predictions


class CausalConvolution(nn.Module):
    '''
        Define a causal convolution without violating autoregressive property. 
        yi cannot depend on xi
    '''
    def __init__(self, in_channels = 1, out_channels = 1, kernel_size = 2, dilation = 1):
        super(CausalConvolution, self).__init__()
        self.padding = (kernel_size - 1) * dilation + 1
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding = self.padding)

    def forward(self, x):
        return self.conv(x)[:,:,:-(self.padding + 1)]


class DilatedCausalConvolution(nn.Module):
    '''
        Define a causal convolution without violating autoregressive property. 
        yi can depend on xi
    '''
    def __init__(self, in_channels = 1, out_channels = 1, kernel_size = 2, dilation = 1):
        super(DilatedCausalConvolution, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels=in_channels, 
                            out_channels=out_channels, 
                            kernel_size=kernel_size, 
                            padding = self.padding)

    def forward(self, x):
        return self.conv(x)[:,:,:-(self.padding)]

class ResBlock(nn.Module):
    '''
        Build a residual block
    '''
    def __init__(self, res_channels = 128, dilation = 1):
        super(ResBlock, self).__init__()
        self.dilation = DilatedCausalConvolution(in_channels = res_channels, out_channels = res_channels, dilation=dilation)
        self.conv1 = nn.Conv1d(res_channels, res_channels, 1)
        self.conv2 = nn.Conv1d(res_channels, res_channels, 1)

    def forward(self, x):
        dilated = self.dilation(x)
        gated = nn.functional.tanh(dilated) * nn.functional.sigmoid(dilated)
        outputs = self.conv1(gated)[:,:,:x.shape[2]] + x
        return self.conv2(outputs)[:,:,:x.shape[2]]
        
class ResStack(nn.Module):
    '''
        Build a res stack by grouping together a list of res block
    '''
    def __init__(self, res_channels=128, num_res_blocks = 3):
        super(ResStack, self).__init__()
        self.res_stack = [ ResBlock(res_channels, dilation=2**i) for i in range(num_res_blocks)]
        self.res_stack = nn.Sequential(*self.res_stack)
        
    def forward(self, x):
        return self.res_stack(x)


class WaveNet(nn.Module):
    '''
    References - https://github.com/golbin/WaveNet/blob/master/wavenet/networks.py
    '''
    def __init__(self):
        super(WaveNet, self).__init__()
        self.net = nn.Sequential(*[ CausalConvolution(in_channels=1, out_channels=128), 
                                    ResStack(res_channels=128, num_res_blocks=2),
                                    nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1)
                                    ])

    def forward(self, x):
        return self.net(x).reshape((-1, x.shape[2]))

    def nll(self):
        pass
    
    def get_sample(self):
        pass

if __name__ == '__main__':
    wave_net = WaveNet()
    random_input = torch.randn((5, 1, 784))
    print(wave_net(random_input).shape)
