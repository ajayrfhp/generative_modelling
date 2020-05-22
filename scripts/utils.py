import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def tensor2image(x):
      return x.detach().numpy().transpose(0, 2, 3, 1)

def image2tensor(x):
  return torch.tensor(x.transpose(0, 3, 1, 2))

def display_image(image):
  plt.figure(figsize=(1,1))
  plt.imshow(image[:,:,0], cmap = 'gray')
  plt.show()


def display_image_side(image1, image2, save = None):
    
    plt.subplot(1, 2, 1)
    plt.imshow(image1[:,:,0], cmap = 'gray')
    plt.subplot(1, 2, 2)
    plt.imshow(image2[:,:,0], cmap = 'gray')
    if save:
      plt.savefig(save)    
    else:
      plt.show()
