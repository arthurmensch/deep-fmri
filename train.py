import functools
import math

import torch
from os.path import expanduser
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary

from data import get_dataset
from model import VAE, masked_mse

batch_size = 48
in_memory = True

train_dataset, test_dataset, mask = get_dataset(in_memory=in_memory)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False)
model = VAE()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
mask = mask.to(device)

summary(model, (1, 91, 109, 91))

loss_function = functools.partial(masked_mse, mask=mask)

optimizer = Adam(model.parameters(), lr=1e-3, amsgrad=True)

n_epochs = 100
total_loss = 0

n_batch = math.ceil(len(train_dataset) / batch_size)

for epoch in range(n_epochs):
    epoch_batch = 0
    verbose_loss = 0
    verbose_penalty = 0
    verbose_batch = 0
    for this_data in train_loader:
        model.train()
        model.zero_grad()
        this_data = this_data.to(device)
        rec, penalty = model(this_data)
        loss = loss_function(rec, this_data)
        elbo = loss + penalty
        elbo.backward()
        optimizer.step()
        verbose_loss += loss.item()
        verbose_penalty += penalty.item()
        epoch_batch += 1
        verbose_batch += 1
        if epoch_batch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_batch = 0
                val_loss = 0
                val_penalty = 0
                for this_test_data in test_loader:
                    this_test_data = this_test_data.to(device)
                    rec, this_val_penalty = model(this_test_data)
                    this_val_loss = loss_function(rec, this_test_data)
                    val_loss += this_val_loss.item()
                    val_penalty += this_val_penalty.item()
                    val_batch += 1
            val_loss /= val_batch
            val_penalty /= val_batch
            verbose_loss /= verbose_batch
            verbose_penalty /= verbose_batch
            print('Epoch %i, batch %i/%i,'
                  'train_objective: %4e,'
                  'train_penalty: %4e,'
                  'val_objective: %4e,'
                  'val_penalty: %4e' % (epoch, epoch_batch, n_batch,
                                        verbose_loss, verbose_penalty,
                                        val_loss, val_penalty))
            verbose_batch = 0
            train_loss = 0
            penalty = 0
    state_dict = model.state_dict()
    name = 'vae_e_%i_loss_%.4e.pkl' % (epoch, verbose_loss)
    torch.save(state_dict, expanduser('~/output/deep-fmri/%s' % name))
