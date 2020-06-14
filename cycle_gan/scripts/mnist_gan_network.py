import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
from torchsummary import summary
import numpy as np
"""**Network code**"""


class Generator(nn.Module):
    def __init__(self):
        '''
        Network is a map from noise vector to image
        (Batch, 128) -> (Batch, 1, w, h)
        '''
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, 2)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)

        self.fc2 = nn.Linear(128, 128)
        self.conv1T = nn.ConvTranspose2d(32, 32, 3, 2)
        self.conv2T = nn.ConvTranspose2d(64, 32, 3, 2)
        self.conv3T = nn.ConvTranspose2d(64, 32, 3, 2)
        self.conv4T = nn.ConvTranspose2d(32, 1, 3, 1, padding=1)

    def forward(self, x):
        # DOWN
        x1 = self.conv1(x)  # (Batch_size, 32, 13, 13)
        x1 = nn.LeakyReLU(0.2)(x1)  # (Batch_size, 32, 13, 13)
        x2 = self.conv2(x1)  # (Batch_size, 32, 6, 6)
        x2 = nn.LeakyReLU(0.2)(x2)  # (Batch_size, 32, 6, 6)
        x3 = self.fc1(x2.reshape((-1, 32 * 6 * 6)))  # (Batch_size, 128)
        x3 = nn.LeakyReLU(0.2)(x3)  # (Batch_size, 128)

        # UP
        x4 = self.fc2(x3)  # (Batch_size, 128)
        x4 = nn.LeakyReLU(0.1)(x4)  # (Batch_size, 128)
        x4 = x4.view((-1, 32, 2, 2))  # (Batch_size, 32, 2, 2)

        # (Batch_size, 32, 6 , 6)
        x5 = self.conv1T(x4, output_size=(x.shape[0], 32, 6, 6))
        x5 = nn.BatchNorm2d(32)(x5)  # (Batch_size, 32 * 6 * 6)
        x5 = nn.LeakyReLU(0.1)(x5)  # (Batch_size, 32, 6, 6)

        x5_stack = torch.cat((x5, x2), dim=1)  # (Batch_size, 64, 6, 6)
        x6 = self.conv2T(x5_stack, output_size=(
            x.shape[0], 32, 13, 13))  # (Batch_size, 32, 13, 13)
        x6 = nn.BatchNorm2d(32)(x6)  # (Batch_size, 32 * 13 * 13)
        x6 = nn.LeakyReLU(0.1)(x6)  # (Batch_size, 32, 13, 13)

        x6_stack = torch.cat((x6, x1), dim=1)  # (Batch_size, 64, 13, 13)
        x7 = self.conv3T(x6_stack, output_size=(
            x.shape[0], 1, 28, 28))  # (Batch_size, 1, 28, 28)
        x7 = nn.LeakyReLU(0.1)(x7)
        # (Batch_size, 1, 28, 28)
        x8 = self.conv4T(x7, output_size=(x.shape[0], 1, 28, 28))

        return nn.Tanh()(x8)


class Discriminator(nn.Module):
    def __init__(self):
        '''
        Network is a map from an image to (0, 1)
        (Batch, 1, w, h) -> (Batch, )
        '''
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, 2)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)  # (Batch_size, 32, 13, 13)
        x = nn.LeakyReLU(0.2)(x)  # (Batch_size, 32, 13, 13)
        x = self.conv2(x)  # (Batch_size, 32, 6, 6)
        x = nn.LeakyReLU(0.2)(x)  # (Batch_size, 32, 6, 6)
        x = x.reshape((-1, 32 * 6 * 6))  # (Batch_size, 32 * 6 * 6)
        x = self.fc1(x)  # (Batch_size, 128)
        x = nn.LeakyReLU(0.2)(x)  # (Batch_size, 128)
        x = self.output(x)
        return nn.Sigmoid()(x)
