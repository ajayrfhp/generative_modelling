# -*- coding: utf-8 -*-
"""
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
''''from torch.utils.tensorboard import SummaryWriter'''
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import utils
from simple_gan_network import Generator, Discriminator
from cycle_gan_utils import train, visualize_predictions, save_model, load_model
from face_dataset import FaceDataset


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.current_device(), torch.cuda.device_count()


train_dataset = FaceDataset('Happy', 'Sad', 'train')
for i in range(5):
    train_dataset.display(i)