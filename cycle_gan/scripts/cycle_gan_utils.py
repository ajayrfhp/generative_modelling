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
  cycle_loss = |F(G(x)) - x | + |G(F(y)) - y |
  G_loss = CE(ones, D_y(G(x))) + cycle_loss
  F_loss = CE(ones, D_x(G(y))) + cycle_loss
  G_loss += cycle_loss
  F_loss += cycle_loss

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
import utils


def train(epoch, G, F, D_X, D_Y, G_optimizer, F_optimizer, D_X_optimizer, D_Y_optimizer, train_loader, writer, test_loader, max_samples_per_batch = np.inf):
    G_losses, F_losses = 0, 0
    D_X_losses, D_Y_losses = 0, 0
    cycle_losses = 0
    for i, (X, Y) in enumerate(train_loader):
        if torch.cuda.is_available():
            X = X.cuda()
            Y = Y.cuda()
        if i >= max_samples_per_batch:
            break
        for p in D_X.parameters():
            p.requires_grad = True
        for p in D_Y.parameters():
            p.requires_grad = True

        # train D_X, D_Y
        D_X.zero_grad()
        D_Y.zero_grad()

        D_Y_loss = nn.BCELoss()(D_Y(Y), torch.ones(
            (X.shape[0], 1))) + nn.BCELoss()(D_Y(G(X)), torch.zeros((X.shape[0], 1)))
        D_X_loss = nn.BCELoss()(D_X(X), torch.ones(
            (X.shape[0], 1))) + nn.BCELoss()(D_X(F(Y)), torch.zeros((X.shape[0], 1)))
        D_Y_loss.backward(retain_graph=True)
        D_X_loss.backward(retain_graph=True)
        D_X_optimizer.step()
        D_Y_optimizer.step()

        # train G, F
        F.zero_grad()
        G.zero_grad()
        for p in D_X.parameters():
            p.requires_grad = False
        for p in D_Y.parameters():
            p.requires_grad = False

        cycle_loss = 10 * torch.mean(
            torch.abs(F(G(X)) - X)) + torch.mean(torch.abs(G(F(Y)) - Y))
        G_loss = nn.BCELoss()(D_Y(G(X)), torch.ones(
            (X.shape[0], 1)))
        G_total_loss =  G_loss + cycle_loss
        
        F_loss = nn.BCELoss()(D_X(F(Y)), torch.ones(
            (X.shape[0], 1))) 
        F_total_loss = F_loss + cycle_loss

        G_total_loss.backward(retain_graph=True)
        F_total_loss.backward()
        G_optimizer.step()
        F_optimizer.step()

        G_losses += G_loss.item()
        F_losses += F_loss.item()
        D_X_losses += D_X_loss.item()
        D_Y_losses += D_Y_loss.item()
        cycle_losses += cycle_loss.item()
        step = epoch * len(train_loader) + i
        if i % 999 == 0:
            writer.add_scalar('D_X_loss', D_X_losses/1000, step)
            writer.add_scalar('D_Y_loss', D_Y_losses/1000, step)
            writer.add_scalar('G_loss', G_losses/1000, step)
            writer.add_scalar('F_loss', F_losses/1000, step)
            writer.add_scalar('cycle_loss', cycle_losses/1000, step)
            G_losses, F_losses = 0, 0
            D_X_losses, D_Y_losses = 0, 0
            cycle_losses = 0
            G.eval()
            visualize_predictions(test_loader, G, step, writer)
            G.train()

def visualize_predictions(test_loader, G, step, writer):
    n_samples = len(test_loader.dataset)

    input_batch_numpy = []
    prediction_batch_numpy = []
    for j in range(10):
        random_index = int(np.random.random()*n_samples)
        inputs, _ = test_loader.dataset[random_index]
        inputs = inputs.unsqueeze(dim=0)
        predictions = G(inputs).cpu()
        inputs_numpy = utils.tensor2image(inputs.cpu())[0]
        predictions_numpy = utils.tensor2image(predictions.cpu())[0]
        inputs_numpy = (inputs_numpy * 0.5 + 0.5) * 255
        predictions_numpy = (predictions_numpy * 0.5 + 0.5) * 255
        figure = utils.display_image_side(inputs_numpy, predictions_numpy)
        writer.add_figure(f'translations_{step}_{j}', figure, global_step=step)


def save_model(G_path, G, F, F_path):
    torch.save(G.state_dict(), G_path)
    torch.save(F.state_dict(), F_path)


def load_model(G_path, F_path, G, F):
    G.load_state_dict(torch.load(G_path))
    F.load_state_dict(torch.load(F_path))
    G.eval()
    F.eval()
    return G, F
