import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from .parse import parse_data, parse_labels

def get_dataset(folderpath, mode='train'):
    xs, ys = parse_data(os.path.join(folderpath, f'{mode}_x.csv'))
    xs_numpy = np.array(xs, dtype=np.float32)
    ys_numpy = np.array(ys, dtype=np.float32)
    data = np.stack([xs_numpy, ys_numpy], axis=-1)
    data = torch.from_numpy(data)

    labels = parse_labels(os.path.join(folderpath, f'{mode}_y.csv'))
    labels = np.array(labels, dtype=np.float32)
    labels = torch.from_numpy(labels)

    return TensorDataset(data, labels)

def get_train_dataloader(folderpath, batch_size):
    dataset = get_dataset(folderpath, 'train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=11)