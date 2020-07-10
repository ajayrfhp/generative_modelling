# -*- coding: utf-8 -*-
"""unsupervised_image_translation_mnist
"""
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import argparse

from utils import utils
from utils.cycle_gan_utils import train, visualize_predictions, save_model, load_model
from models.cycle_gan_network import Generator, Discriminator


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.current_device(), torch.cuda.device_count()


def training_mode(model_name):
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
    
    # Apply transformation 
    X = np.array([utils.tensor2image(image_batch)
                for image_batch, _ in train_loader])
    X = X.reshape((X.shape[0] * X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
    Y = np.array(X) * -1
    np.random.shuffle(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    unsupervised_train_loader = data.DataLoader(data.TensorDataset(
        utils.image2tensor(X_train), utils.image2tensor(Y_train)), batch_size=1)
    unsupervised_test_loader = data.DataLoader(data.TensorDataset(
        utils.image2tensor(X_test), utils.image2tensor(Y_test)), batch_size=1)

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
    for epoch in range(1):
        train(epoch, G, F, D_X, D_Y, G_optimizer, F_optimizer, D_X_optimizer,
            D_Y_optimizer, unsupervised_train_loader, writer, test_loader)

    save_model(f'../saved_models/{model_name}_G.pt', G, F, f'../saved_models/{model_name}_F.pt')
    

def testing_mode(model_name):
    G, F = load_model(f'../saved_models/{model_name}_G.pt', f'../saved_models/{model_name}_F.pt', G, F)
    visualize_predictions(test_loader, G, 1 * len(train_loader) + 5, writer)


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
