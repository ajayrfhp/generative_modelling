"""**Psuedocode**

 Networks D_x, D_y, F, G

```
G maps from X to Y
F maps from Y to X

D_x maps from (b, w_x, s_x, 3) -> {0, 1}
D_y maps from (b, w_y, s_y, 3) -> {0, 1}

G maps from (b, w_x, s_x, 3) -> (b, w_y, s_y, 3)
F maps from (b, w_y, s_y, 3) -> (b, w_x, s_x, 3)

CE(y, y_pred) = -sum(ylog(y_pred))

for batch X, Y
  D_y_loss = CE(ones, D_y(y)) + CE(zeros, D_y(G(x)))
  D_x_loss = CE(ones, D_x(x)) + CE(zeros, D_x(G(y)))
  cyclical_loss = |F(G(x)) - x | + |G(F(y)) - y |
  G_loss = CE(ones, D_y(G(x))) + cyclical_loss
  F_loss = CE(ones, D_x(G(y))) + cyclical_loss
  G_loss += cyclical_loss
  F_loss += cyclical_loss

  Compute gradients
  Update G, F, D_x, D_y weights

```

**Write Objective function, training loop**
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from simple_gan_network import Generator, Discriminator
import utils

def train(epoch, G, F, D_X, D_Y, G_optimizer, F_optimizer, D_X_optimizer, D_Y_optimizer, train_loader, writer, test_loader):
    G_losses, F_losses = 0, 0
    D_X_losses, D_Y_losses = 0, 0
    for i, (X, Y) in enumerate(train_loader):
        if torch.cuda.is_available():
            X = X.cuda()
            Y = Y.cuda()
        for p in D_X.parameters(): p.requires_grad = True
        for p in D_Y.parameters(): p.requires_grad = True

        # train D_X, D_Y
        D_X.zero_grad()
        D_Y.zero_grad()

        D_Y_loss = nn.BCELoss()(D_Y(Y), torch.ones((X.shape[0], 1))) + nn.BCELoss()(D_Y(G(X)), torch.zeros((X.shape[0], 1)))
        D_X_loss = nn.BCELoss()(D_X(X), torch.ones((X.shape[0], 1))) + nn.BCELoss()(D_X(F(Y)), torch.zeros((X.shape[0], 1)))
        D_Y_loss.backward(retain_graph=True)
        D_X_loss.backward(retain_graph=True)
        D_X_optimizer.step()
        D_Y_optimizer.step()

        # train G, F
        F.zero_grad()
        G.zero_grad()
        for p in D_X.parameters(): p.requires_grad = False
        for p in D_Y.parameters(): p.requires_grad = False

        cyclical_loss = torch.mean(torch.abs(F(G(X)) - X)) + torch.mean(torch.abs(G(F(Y)) - Y))
        
        G_loss = nn.BCELoss()(D_Y(G(X)), torch.ones((X.shape[0], 1))) + 2 * cyclical_loss
        F_loss = nn.BCELoss()(D_X(F(Y)), torch.ones((X.shape[0], 1))) + 2 * cyclical_loss

        G_loss.backward(retain_graph = True)
        F_loss.backward()
        G_optimizer.step()
        F_optimizer.step()
        
        G_losses += G_loss.item()
        F_losses += F_loss.item()
        D_X_losses += D_X_loss.item()
        D_Y_losses += D_Y_loss.item()

        if i % 4 == 0:
            writer.add_scalar('D_X_loss', D_X_losses/5, epoch * 100 + i)
            writer.add_scalar('D_Y_loss', D_Y_losses/5, epoch * 100 + i)
            writer.add_scalar('G_loss', G_losses/5, epoch * 100 + i)
            writer.add_scalar('F_loss', F_losses/5, epoch * 100 + i)
            G_losses, F_losses = 0, 0
            D_X_losses, D_Y_losses = 0, 0
            visualize_predictions(test_loader, G, epoch * 100 + i, writer)

        if i > 100:
            return 

def visualize_predictions(test_loader, G, step, writer):
    n_samples = len(test_loader.dataset)

    input_batch_numpy = []
    prediction_batch_numpy = []
    for _ in range(10):
        random_index = int(np.random.random()*n_samples)
        inputs, _ = test_loader.dataset[random_index]
        inputs = inputs.unsqueeze(dim=0)
        predictions = G(inputs).cpu()
        inputs_numpy = utils.tensor2image(inputs.cpu())
        predictions_numpy = utils.tensor2image(predictions.cpu())
        input_batch_numpy.append(inputs_numpy[0])
        prediction_batch_numpy.append(predictions_numpy[0])
    input_batch_numpy = np.array(input_batch_numpy)
    prediction_batch_numpy = np.array(prediction_batch_numpy)
    
    writer.add_images('inputs', input_batch_numpy, global_step = step, dataformats = 'NHWC')
    writer.add_images('predictions', prediction_batch_numpy, global_step = step, dataformats = 'NHWC')

def save_model(G_path, G, F, F_path):
    torch.save(G.state_dict(), G_path)
    torch.save(F.state_dict(), F_path)

def load_model(G_path, F_path):
    G = Generator()
    F = Generator()
    G.load_state_dict(torch.load(G_path))
    F.load_state_dict(torch.load(F_path))
    return G, F
