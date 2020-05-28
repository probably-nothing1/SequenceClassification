import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from .parse import parse_data, parse_labels, parse_and_preprocess_data

def get_dataset(folderpath, mode='train'):
    xs, ys = parse_data(os.path.join(folderpath, f'{mode}_x.csv'))
    xs_numpy = np.array(xs, dtype=np.float32)
    ys_numpy = np.array(ys, dtype=np.float32)
    data = np.stack([xs_numpy, ys_numpy], axis=-1)
    data = torch.from_numpy(data)

    labels = parse_labels(os.path.join(folderpath, f'{mode}_y.csv'))
    labels = np.array(labels)
    labels = torch.from_numpy(labels)

    return TensorDataset(data, labels)

def get_preprocessed_dataset(folderpath, mode='train'):
    xs = parse_and_preprocess_data(os.path.join(folderpath, f'{mode}_x.csv'))
    # xs_numpy = np.array(xs)
    data = torch.LongTensor(xs)

    labels = parse_labels(os.path.join(folderpath, f'{mode}_y.csv'))
    labels = np.array(labels)
    labels = torch.from_numpy(labels)

    return TensorDataset(data, labels)

def get_train_dataloader(folderpath, batch_size, embedding=False):
    if embedding:
        dataset = get_preprocessed_dataset(folderpath, 'train')
    else:
        dataset = get_dataset(folderpath, 'train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=11)

def get_test_dataloader(folderpath, batch_size, embedding=False):
    if embedding:
        dataset = get_preprocessed_dataset(folderpath, 'test')
    else:
        dataset = get_dataset(folderpath, 'test')
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=11)
