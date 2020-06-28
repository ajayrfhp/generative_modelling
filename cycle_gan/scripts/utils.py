import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
plt.rcParams.update({'font.size': 30})

def tensor2image(x):
    if len(list(x.size())) == 4:
        return x.detach().numpy().transpose(0, 2, 3, 1)
    return x.detach().numpy().transpose(1, 2, 0)


def image2tensor(x):
    if len(x.shape) == 4:
        return torch.tensor(x.transpose(0, 3, 1, 2))
    if len(x.shape) == 3:
        return torch.tensor(x.transpose(2, 0, 1))
    return torch.tensor(x).unsqueeze(dim=0)

def display_images(images):
    titles = ["inputs", "predictions", "mask", "copied", "synthesized", "weighted_synthesized"]
    fig, axes = plt.subplots(nrows = 1, ncols = len(images), figsize = (100, 30))
    for ax, image, title in zip(axes, images, titles):
        ax.set_title(title)
        ax.imshow(image.astype(np.uint8))
    return fig
        

def display_image(image):
    if image.shape[2] == 1:
        plt.figure(figsize=(1, 1))
        plt.imshow(image[:, :, 0], cmap='gray')
        plt.show()
    else:
        plt.figure(figsize=(1, 1))
        plt.imshow(image[:, :])
        plt.show()


def display_image_side(image1, image2, show=False):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    if image1.shape[-1] == 1:
        axes[0].imshow(image1[:, :, 0], cmap='gray')
    else:
        axes[0].imshow(image1[:, :].astype(np.uint8))
    if image2.shape[-1] == 1:
        axes[1].imshow(image2[:, :, 0], cmap='gray')
    else:
        axes[1].imshow(image2[:, :].astype(np.uint8))
    fig.tight_layout()
    if show:
        plt.show()
    return fig
