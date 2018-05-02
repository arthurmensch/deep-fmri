import math

import torch
import torch.nn.functional as F
from os.path import expanduser
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import get_dataset
from model import VAE

train_dataset, test_dataset = get_dataset(in_memory=True)

train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32,
                         shuffle=False)
model = VAE()
model = torch.save(state_dict, expanduser('~/output/deep-fmri/%s' % name))
