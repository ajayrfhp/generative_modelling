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

class CausalConv(nn.Conv2d):
    '''
        Output x_i depends on x1..x1-1. 
        Mask Type A is Strictly autoregressive, suitable for input layers
        Mask Type B can depend on input itself, suitable for intermediate layers
    '''
    def __init__(self, mask_type = 'B', in_channels = 1, out_channels = 1, kernel_size = 3, padding = 1):
        super(CausalConv, self).__init__(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding)
        self.register_buffer('mask', torch.ones((out_channels, kernel_size, kernel_size)))

        for i in range(kernel_size):
            for j in range(kernel_size):
                if mask_type == 'A':
                    if i >= kernel_size / 2  or (i == kernel_size // 2 and j >= kernel_size // 2 ):
                        self.mask[:, i, j] = 0
                else:
                    if i >= kernel_size / 2  or (i == kernel_size // 2 and j > kernel_size // 2 ):
                        self.mask[:, i, j] = 0

    def forward(self, input):
        self.weight.data *= self.mask
        return super(CausalConv, self).forward(input)
        
class ConvBlock(nn.Module):
    def __init__(self, mask_type='B',in_channels=1, out_channels=1, kernel_size=3, padding=1, unit_test = False):
        super(ConvBlock, self).__init__()
        self.conv = CausalConv(mask_type, in_channels, out_channels, kernel_size, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(*[self.conv,])
        if unit_test:
            self.conv.weight.data.fill_(1)
            self.conv.bias.data.fill_(0)


    def forward(self, input):
        return self.net(input)


class PixelCNN(nn.Module):
    '''
    References - https://github.com/singh-hrituraj/PixelCNN-Pytorch
    '''
    def __init__(self, in_channels = 1, hidden_channels = 64, out_channels = 1, num_conv_blocks = 5, kernel_size = 3, nin = 784):
        super(PixelCNN, self).__init__()
        self.input_conv = CausalConvA(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.conv_blocks = [ CasualConvB(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size) for _ in range(num_conv_blocks)]
        self.output_conv = CasualConvB(in_channels=hidden_channels, out_channels=out_channels, kernel_size = kernel_size)
        self.net = nn.Sequential(*([self.input_conv] + self.conv_blocks + [self.output_conv]))
        self.nin = nin

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x)
    
    def nll(self, input):
        x = input.unsqueeze(1)
        predictions = self.net(x)
        return nn.BCEWithLogitsLoss()(predictions, x)
    
    def get_sample(self):
        dim = int(np.sqrt(self.nin))
        prediction = torch.zeros((1, 1, dim, dim))
        for i in range(dim):
            for j in range(dim):
                probability = torch.sigmoid(self.net(prediction))
                prediction[0, 0, i, j] = torch.bernoulli(probability[0, 0, i, j])
        return prediction.squeeze(0).squeeze(0)


if __name__ == '__main__':
    # Unit test Mask Type B
    # Input  Output
    # 0 1 2  0 + 1 + 2 + 3 + 4 = 10
    # 3 4 5
    # 6 7 8
    conv_block_b = ConvBlock(padding = 0, unit_test = True)
    x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).unsqueeze(0).unsqueeze(0).to(torch.float)
    assert(conv_block_b(x)[0][0][0][0].item() == 10)

    # Unit test Mask Type A
    # Input  Output
    # 0 1 2  0 + 1 + 2 + 3 = 6
    # 3 4 5
    # 6 7 8
    conv_block_a = ConvBlock(mask_type = "A", padding = 0, unit_test = True)
    x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).unsqueeze(0).unsqueeze(0).to(torch.float)
    assert(conv_block_a(x)[0][0][0][0].item() == 6)


    # Unit test Mask Type B
    # Input  Output
    # 9 1 2  9 + 1 + 2 + 3 + -1 = 14
    # 3 -1 5
    # 6 7111 800000
    conv_block_b = ConvBlock(padding = 0, unit_test = True)
    x = torch.tensor([[9, 1, 2], [3, -1, 5], [6, 7111, 800000]]).unsqueeze(0).unsqueeze(0).to(torch.float)
    assert(conv_block_b(x)[0][0][0][0].item() == 14)