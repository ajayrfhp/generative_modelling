def train(G, F, D_X, D_Y, train_loader):
      G_losses = []
  F_losses = []
  D_Y_losses = []
  D_X_losses = []
  for i, (X, Y) in enumerate(train_loader):
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
    
    G_losses.append(G_loss.mean().item())
    F_losses.append(F_loss.mean().item())
    D_X_losses.append(D_X_loss.mean().item())
    D_Y_losses.append(D_Y_loss.mean().item())

    if i % 100 == 0:
      print(D_X_loss.mean().item(), D_Y_loss.mean().item(), G_loss.mean().item(), F_loss.mean().item(), cyclical_loss.item())
