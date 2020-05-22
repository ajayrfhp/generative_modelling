# -*- coding: utf-8 -*-
"""unsupervised_image_translation_mnist

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lgGhq9JYdG1BGkTanKoGk2MY_vorY9QH

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import utils
from simple_gan_network import Generator, Discriminator
from cycle_gan_utils import train, visualize_predictions, save_model, load_model
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.current_device(), torch.cuda.device_count()

# Load raw data

batch_size = 25

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])),
    batch_size=batch_size, shuffle=True)


# Prepare unpaired image translation dataset


X = np.array([ utils.tensor2image(image_batch) for image_batch, _ in train_loader])
X = X.reshape((X.shape[0] * X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
Y = np.array(X) * -1
np.random.shuffle(Y)

#utils.display_image(X[0])
#utils.display_image(Y[0])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
unsupervised_train_loader = data.DataLoader(data.TensorDataset(utils.image2tensor(X_train), utils.image2tensor(Y_train)), batch_size=1)
unsupervised_test_loader = data.DataLoader(data.TensorDataset(utils.image2tensor(X_test), utils.image2tensor(Y_test)), batch_size=1)

x, y = next(iter(unsupervised_train_loader))
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
writer = SummaryWriter()

# Train network

for epoch in range(1):
  train(epoch, G, F, D_X, D_Y, G_optimizer, F_optimizer, D_X_optimizer, D_Y_optimizer, unsupervised_train_loader, writer, test_loader)


save_model('../models/simple_gan_mnist_G.pt', G, F, '../models/simple_gan_mnist_F.pt')
G, F = load_model('../models/simple_gan_mnist_G.pt', '../models/simple_gan_mnist_F.pt')


visualize_predictions(test_loader, G, 1 * 100 + 5, writer)



