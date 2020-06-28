# -*- coding: utf-8 -*-
"""
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import utils
from mask_gan_network import Generator, Discriminator
from cycle_gan_utils import train, visualize_predictions, save_model, load_model
from celeba_dataset import CelebADataset
import argparse


def training_mode(model_name):
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.current_device(), torch.cuda.device_count()
    dataset = CelebADataset('../data/celebA/')
    indices = list(range(len(dataset)))
    split = int(0.8 * len(dataset))
    train_indices, test_indices = indices[:split], indices[split:]

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=1, sampler=SubsetRandomSampler(train_indices))
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1, sampler=SubsetRandomSampler(test_indices))

    # Initialize network and optimizers

    G = Generator(3)
    F = Generator(3)
    D_X = Discriminator(3)
    D_Y = Discriminator(3)
    
    G_optimizer = optim.Adam(G.parameters(), lr=1e-3)
    F_optimizer = optim.Adam(F.parameters(), lr=1e-3)
    D_X_optimizer = optim.Adam(D_X.parameters(), lr=1e-3)
    D_Y_optimizer = optim.Adam(D_Y.parameters(), lr=1e-3)
    writer = SummaryWriter(model_name)

    # Train network
    for epoch in range(5):
        print('epoch ', epoch)
        train(epoch, G, F, D_X, D_Y, G_optimizer, F_optimizer,
              D_X_optimizer, D_Y_optimizer, train_loader, writer, test_loader)

    save_model(f'../models/{model_name}_G.pt', G, F, f'../models/{model_name}_F.pt')


def testing_mode(model_name):
    G = Generator(3)
    D = Discriminator(3)
    G, F = load_model(f'../models/{model_name}_G.pt', f'../models/{model_name}_F.pt', G, F)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.current_device(), torch.cuda.device_count()
    dataset = CelebADataset('../data/celebA/')
    indices = list(range(len(dataset)))
    split = int(0.8 * len(dataset))
    train_indices, test_indices = indices[:split], indices[split:]
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1, sampler=SubsetRandomSampler(test_indices))
    writer = SummaryWriter(model_name)
    visualize_predictions(test_loader, G, 0, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Provide information about model and mode')
    parser.add_argument('--model_name', type=str, dest='model_name',
                        help='model name to associate with tensorboard')
    parser.add_argument('--mode', type=str, dest='mode', help='train/test')
    parsed = parser.parse_args()
    model_name = parsed.model_name
    mode = parsed.mode
    if mode == 'train':
        training_mode(model_name)
    if mode == 'test':
        testing_mode(model_name)
