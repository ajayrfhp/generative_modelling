# -*- coding: utf-8 -*-
"""
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import matplotlib
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
test_dataset = FaceDataset('Happy', 'Sad', 'val')

train_loader = data.DataLoader(train_dataset)
test_loader = data.DataLoader(test_dataset)



x, y = next(iter(train_loader))
c = utils.tensor2image(x.cpu()[0:1])
#utils.display_image(c[0])


# Initialize network and optimizers

G = Generator()
F = Generator()
D_X = Discriminator()
D_Y = Discriminator()

G_optimizer = optim.Adam(G.parameters(), lr = 1e-4)
F_optimizer = optim.Adam(F.parameters(), lr = 1e-4)
D_X_optimizer = optim.Adam(D_X.parameters(), lr = 1e-4)
D_Y_optimizer = optim.Adam(D_Y.parameters(), lr = 1e-4)
now = datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter('cycle_gan/' + now) 

# Train network
for epoch in range(5):
  train(epoch, G, F, D_X, D_Y, G_optimizer, F_optimizer, D_X_optimizer, D_Y_optimizer, train_loader, writer, test_loader)


save_model('../models/cycle_gan_happy_sad_G.pt', G, F, '../models/cycle_gan_happy_sad_F.pt')
G, F = load_model('../models/cycle_gan_happy_sad_G.pt', '../models/cycle_gan_happy_sad_F.pt')


visualize_predictions(test_loader, G, 1 * 100 + 5, writer)

