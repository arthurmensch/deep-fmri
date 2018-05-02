import torch
from os.path import expanduser
from torch.utils.data import DataLoader

from data import get_dataset
from model import VAE

train_dataset, test_dataset = get_dataset(in_memory=True)

train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32,
                         shuffle=False)
model = VAE()
name = 'vae_e_42_loss_1.1013e+06.pkl'
model = torch.load(expanduser('~/output/deep-fmri/%s' % name))

data = test_dataset[0]
rec = model(data)

