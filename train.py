import math

import torch

import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary

from data import get_dataset
from model import VAE

dataset = get_dataset()
loader = DataLoader(dataset, batch_size=32,
                    shuffle=True)
model = VAE()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
# print(summary(model, (1, 91, 109, 91)))

optimizer = Adam(model.parameters(), amsgrad=True)

n_epochs = 10
total_loss = 0

n_batch = math.ceil(len(dataset) / 32)

for epoch in range(n_epochs):
    epoch_batch = 0
    train_loss = 0
    penalty = 0
    verbose_batch = 0
    for this_data in loader:
        this_data = this_data.to(device)
        model.zero_grad()
        rec, penalty = model(this_data)
        loss = F.mse_loss(rec, this_data)
        loss += penalty
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        penalty += penalty.item()
        epoch_batch += 1
        verbose_batch += 1
        if epoch_batch % 10 == 0:
            train_loss /= verbose_batch
            penalty /= verbose_batch
            print('Epoch %i, batch %i/%i,'
                  ' train_objective: %.4f,'
                  'penalty: %.4f' % (epoch, epoch_batch,
                     train_loss, penalty))
            verbose_batch = 0
            train_loss = 0
            penalty = 0
