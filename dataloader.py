

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from imblearn.over_sampling import RandomOverSampler

from torch.utils.data.dataset import random_split


from collections import Counter

from torch.utils.data import SubsetRandomSampler

def balanced_random_split(dataset, lengths):
    """
    Args:
      dataset: A torch.utils.data.Dataset instance.
      lengths: A list of ints, specifying the lengths of the splits to make.
    Returns:
      A list of torch.utils.data.SubsetRandomSampler instances, one for each split.
    """
    # Get the number of items in the dataset
    n = len(dataset)

    # get index of first instance of class 1
    m = int(n/2)

    class_0_idx = list(range(m))
    class_1_idx = list(range(m, n))
    # Create a list of indices for the dataset

    c0_lengths = [int(l/2) for l in lengths]
    c1_lengths = [l - c0 for l, c0 in zip(lengths, c0_lengths)]
    c0_split_indices = [class_0_idx[i:i + l] for i, l in enumerate(c0_lengths)]
    c1_split_indices = [class_1_idx[i:i + l] for i, l in enumerate(c1_lengths)]



    # combine the indices 
    split_indices = [c0 + c1 for c0, c1 in zip(c0_split_indices, c1_split_indices)]

    # shuffle the indices
    for i in range(len(split_indices)):
        np.random.shuffle(split_indices[i])

    # create a list of subset samplers
    samplers = [Subset(dataset, indices) for indices in split_indices]

    return samplers

def balanced_random_split_v2(dataset, lengths, num_classes=4):
    """
    Args:
      dataset: A torch.utils.data.Dataset instance, assumed to have an equally balanced class distribution.
      lengths: A list of ints, specifying the lengths of the splits to make.
      num_classes: The number of classes in the dataset.
    Returns:
      A list of torch.utils.data.SubsetRandomSampler instances, one for each split.
    """
    # Get the number of items in the dataset
    n = len(dataset)
    
    # Calculate the number of samples per class
    n_per_class = int(n / num_classes)




class FCMatrixDataset(Dataset):
    def __init__(self, ukb_filtered, dir, data_field, mapping, oversample=False):
        print(ukb_filtered)
        self.ukb_filtered = pd.read_csv(ukb_filtered, sep=' ')
        self.dir = dir
        self.data_field = data_field
        if mapping is not None:
            self.mapping ={'HC': 0, 'single ep.': 1, 'moderate rMDD': 2, 'severe rMDD': 3}
        else:
            self.mapping = [0, 1]

        if oversample:
            ros = RandomOverSampler(random_state=0)


    def __len__(self):
        return len(self.ukb_filtered)

    def __getitem__(self, idx):
        eid = self.ukb_filtered.iloc[idx, 0]
        filename = str(eid) + '_' + self.data_field + '_2_0.txt'
        matrix_path = os.path.join(self.dir, filename) 
        label = self.ukb_filtered.iloc[idx, 1]
        # label HC, single ep., moderate rMDD, severe rMDD
        enc_label = self.mapping[label]

        with open(matrix_path, "r") as f:
            matrix = f.read().split()
        matrix = [float(item) for item in matrix]
        matrix = torch.FloatTensor(matrix)
        return matrix, enc_label

if __name__ == "__main__":
    # ds = FCMatrixDataset('data/ukb_filtered_25753_harir_mh_upto69.csv','data/fetched/25753', '25753', 1)
    # ds = FCMatrixDataset('data/gal_eids/gal_data.csv','data/fetched/25753_gal', '25753', None)
    ds = FCMatrixDataset('data/fully_balanced_ukb_filtered_25753_harir_mh_upto69.csv','data/fetched/25751', '25751', 1)
    total_size = len(ds)
    print(total_size)
    train_size = int(.8* total_size)
    val_size = int(.1 * total_size)
    test_size = total_size - train_size - val_size

    train, val, test = balanced_random_split_v2(ds, [train_size, val_size, test_size])

    batch_size = 128

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)


    all_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    train_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    val_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    test_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for i, (inputs, targets) in enumerate(train_loader):
        for t in targets:
            train_counts[t.item()] += 1
            all_counts[t.item()] += 1
    
    for i, (inputs, targets) in enumerate(val_loader):
        for t in targets:
            val_counts[t.item()] += 1
            all_counts[t.item()] += 1

    for i, (inputs, targets) in enumerate(test_loader):
        for t in targets:
            test_counts[t.item()] += 1
            all_counts[t.item()] += 1

    print(all_counts)
    print(train_counts)
    print(val_counts)
    print(test_counts)

