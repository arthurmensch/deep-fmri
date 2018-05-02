import glob

import numpy as np
import torch
from os.path import expanduser, join
from torch.utils.data import Dataset, ConcatDataset


class NumpyDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename

    def __len__(self):
        return np.load(self.filename, mmap_mode='r').shape[3]

    def __getitem__(self, index):
        data = np.load(self.filename, mmap_mode='r')
        return torch.Tensor(data[:, :, :, index])


def get_dataset(subject=100307, output_dir=None):
    if output_dir is None:
        output_dir = expanduser('~/data/HCP_masked')
    datasets = []
    for filename in glob.glob(join(output_dir, '%s_REST*.npy' % subject)):
        datasets.append(NumpyDataset(filename))
    dataset = ConcatDataset(datasets)
    return dataset

