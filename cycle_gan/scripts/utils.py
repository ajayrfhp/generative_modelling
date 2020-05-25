import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def tensor2image(x):
      if len(list(x.size())) == 4:
            return x.detach().numpy().transpose(0, 2, 3, 1)
      return  x.detach().numpy().transpose(1, 2, 0)


def image2tensor(x):
      if len(x.shape) == 4:
            return torch.tensor(x.transpose(0, 3, 1, 2))
      if len(x.shape) == 3:
            return torch.tensor(x.transpose(2, 0, 1))
      return torch.tensor(x).unsqueeze(dim = 0)      

def display_image(image):
      plt.figure(figsize=(1,1))
      plt.imshow(image[:,:,0], cmap = 'gray')
      plt.show()


def display_image_side(image1, image2, show = False):
      fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
      axes[0].imshow(image1[:,:,0], cmap = 'gray')
      axes[1].imshow(image2[:,:,0], cmap = 'gray')
      fig.tight_layout()
      if show:
            plt.show()
      return fig

def compute_means_stds(train_loader, num_samples = 1000):
      means = 0
      stds = 0
      for i, (input_sample, output_sample) in enumerate(train_loader):
            if i > num_samples:
                  break
            means += input_sample.mean().item() / num_samples
            stds +=  input_sample.std().item() / num_samples
      return means, stds
