import glob

import numpy as np
import torch
from os.path import expanduser, join
from skimage.morphology import dilation, binary_dilation
from torch.utils.data import Dataset, ConcatDataset


class NumpyDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename

    def __len__(self):
        return np.load(self.filename, mmap_mode='r').shape[3]

    def __getitem__(self, index):
        data = np.load(self.filename, mmap_mode='r')
        return torch.Tensor(data[None, :, :, :, index])


class NumpyDatasetMem(Dataset):
    def __init__(self, filename):
        self.data = torch.Tensor(np.load(filename,
                                         mmap_mode=None))

    def __len__(self):
        return self.data.shape[3]

    def __getitem__(self, index):
        return self.data[None, :, :, :, index]


def get_dataset(subject=100307, output_dir=None, in_memory=False):
    if in_memory:
        dataset_type = NumpyDatasetMem
    else:
        dataset_type = NumpyDataset
    if output_dir is None:
        output_dir = expanduser('~/data/HCP_masked')
    datasets = []
    for filename in glob.glob(join(output_dir, '%s_REST*.npy' % subject)):
        datasets.append(dataset_type(filename))
    train_dataset = ConcatDataset(datasets[:-1])
    test_dataset = datasets[-1]
    mask = np.load(expanduser('~/data/HCP_masked/%s_mask.npy'
                              % subject))
    mask = mask.astype('bool')
    print('Mask', mask.astype('float').sum(), 'voxels')
    for i in range(2):
        mask = binary_dilation(mask)
    mask = mask.astype('uint8')
    print('Dilated mask', mask.astype('float').sum(), 'voxels')
    mask = torch.from_numpy(mask).byte()

    return train_dataset, test_dataset, mask

