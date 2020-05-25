import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
import numpy as np
"""**Network code**"""

class Generator(nn.Module):
  def __init__(self):
    '''
    Network is a map from image to image
    (Batch, 1, 48, 48) -> (Batch, 1, 48, 48)
    '''
    super(Generator, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 2)
    self.conv2 = nn.Conv2d(32, 32, 3, 2)
    self.conv3 = nn.Conv2d(32, 32, 3, 2)
    self.fc1 = nn.Linear(32 * 5 * 5, 128)
    self.fc2 = nn.Linear(128, 32 * 5 * 5)
    self.conv1T = nn.ConvTranspose2d(32, 32, 3, 2)
    self.conv2T = nn.ConvTranspose2d(64, 32, 3, 2)
    self.conv3T = nn.ConvTranspose2d(64, 1, 3, 2)

    
  def forward(self, x):
    # DOWN 
    x1 = self.conv1(x) # (Batch_size, 32, 23, 23)
    x1 = nn.LeakyReLU(0.2)(x1) # (Batch_size, 32, 23, 23)
    x2 = self.conv2(x1) # (Batch_size, 32, 11, 11)
    x2 = nn.LeakyReLU(0.2)(x2) # (Batch_size, 32, 11, 11)
    x3 = self.conv3(x2) # (Batch_size, 32, 5, 5)
    x3 = nn.LeakyReLU(0.2)(x3) # (Batch_size, 32, 5, 5)
    x4 = self.fc1(x3.reshape((-1, 32 * 5 * 5))) # (Batch_size, 32 * 5 * 5)
    x4 = nn.LeakyReLU(0.2)(x4) # (Batch_size, 128)

    # UP
    x5 = self.fc2(x4) # (Batch_size, 32 * 5 * 5)
    x5 = nn.LeakyReLU(0.1)(x5) # (Batch_size, 32 * 5 * 5)
    x5 = x5.view((-1, 32, 5, 5)) # (Batch_size, 32, 5, 5)
    
    x6 = self.conv1T(x5, output_size = (x.shape[0], 32, 11, 11)) # (Batch_size, 32, 11 , 11)
    x6 = nn.BatchNorm2d(32)(x6) # (Batch_size, 32, 11 , 11)
    x6 = nn.LeakyReLU(0.1)(x6) # (Batch_size, 32, 11 , 11)
    
    x6_stack = torch.cat((x6, x2), dim = 1) ## (Batch_size, 64, 11, 11)
    x7 = self.conv2T(x6_stack, output_size = (x.shape[0], 32 , 23, 23)) # (Batch_size, 32, 23, 23)
    x7 = nn.BatchNorm2d(32)(x7) # (Batch_size, 32 * 23 * 23)
    x7 = nn.LeakyReLU(0.1)(x7) # (Batch_size, 32, 23, 23)

    x8_stack = torch.cat((x7, x1), dim = 1) #(Batch_size, 64, 23, 23)
    x9 = self.conv3T(x8_stack, output_size = (x.shape[0], 1 , 48, 48)) # (Batch_size, 1, 48, 48)
    
    return nn.Tanh()(x9)
    

class Discriminator(nn.Module):
  def __init__(self):
    '''
    Network is a map from an image to (0, 1)
    (Batch, 1, w, h) -> (Batch, )
    '''
    super(Discriminator, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 2)
    self.conv2 = nn.Conv2d(32, 32, 3, 2)
    self.conv3 = nn.Conv2d(32, 32, 3, 2)
    self.fc1 = nn.Linear(32 * 5 * 5, 128)
    self.fc2 = nn.Linear(128, 1)

  def forward(self, x):
    x1 = self.conv1(x) # (Batch_size, 32, 23, 23)
    x1 = nn.LeakyReLU(0.2)(x1) # (Batch_size, 32, 23, 23)
    x2 = self.conv2(x1) # (Batch_size, 32, 11, 11)
    x2 = nn.LeakyReLU(0.2)(x2) # (Batch_size, 32, 11, 11)
    x3 = self.conv3(x2) # (Batch_size, 32, 5, 5)
    x3 = nn.LeakyReLU(0.2)(x3) # (Batch_size, 32, 5, 5)
    x4 = self.fc1(x3.reshape((-1, 32 * 5 * 5))) # (Batch_size, 128)
    x4 = nn.LeakyReLU(0.2)(x4) # (Batch_size, 128)
    x5 = self.fc2(x4) # (Batch_size, 1)
    return nn.Sigmoid()(x5)


'''
F = Generator()
D = Discriminator()
pred = F(torch.randn(5, 1, 48, 48))
pred2 = D(pred)
print(pred2.shape)
'''