

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset



# from imblearn.over_sampling import RandomOverSampler

from torch.utils.data.dataset import random_split
# from utils import balanced_random_split_v2, compute_KNN_graph, fc_to_matrix, adjacency

from collections import Counter

from torch.utils.data import SubsetRandomSampler



class FCMatrixDataset(Dataset):
    def __init__(self, ukb_filtered, dir, data_field, mapping, oversample=False, mrmr=None):
        # self.ukb_filtered = pd.read_csv(ukb_filtered, sep=' ', header=None)
        self.ukb_filtered = np.genfromtxt(ukb_filtered, delimiter= ' ', dtype=int)
        self.dir = dir
        self.data_field = data_field
        self.mapping = mapping
        self.mrmr = mrmr

        if oversample:
            # ros = RandomOverSampler(random_state=0)
            pass


    def __len__(self):
        return len(self.ukb_filtered)

    def __getitem__(self, idx):
        eid = self.ukb_filtered[:,0]

        eid = eid[idx]
        filename = str(eid) + '_' + self.data_field + '_2_0.txt'
        print(filename)

        matrix_path = os.path.join(self.dir, filename)
        label = self.ukb_filtered[:,1][idx]
        if self.mapping is not None:
            # label HC, single ep., moderate rMDD, severe rMDD
            enc_label = self.mapping[label]
        else:
            enc_label = torch.tensor(label).long()
        

        with open(matrix_path, "r") as f:
            matrix = f.read().split()
        matrix = np.array([float(item) for item in matrix])
        print(f"idx 1140 {matrix[1140]}")
        print(f"idx 536 {matrix[689]}")
        if self.mrmr is not None:
            matrix = matrix[self.mrmr]

        print(f"idx 0 {matrix[0]}") 
        print(f"idx 1 {matrix[1]}")
        matrix = torch.FloatTensor(matrix)
        return matrix, enc_label




if __name__ == "__main__":

    # mrmr_features = np.array([1140, 536, 223, 907, 1449, 499, 1293, 45, 135, 1440, 879, 1384, 1210, 1316, 122, 22, 492, 638, 765, 1027, 1464, 501, 1462, 395, 26, 1079, 70, 425, 1403, 1409, 1318, 886, 1459, 1448, 939, 1163, 547, 10, 413, 676, 131, 216, 942, 1136, 1386, 232, 1455, 1337, 814, 139, 392, 1376, 1382, 471, 656]
                             
    mrmr_features = np.array([1140, 689, 427, 122, 139, 765, 907, 1384, 22, 1293, 1449, 492, 1440, 499, 1316, 1318, 135, 879, 886, 223, 1455, 676, 1136, 1464, 70, 1462, 1386, 939, 111, 413, 10, 1403, 1027, 547, 395, 1210, 942, 501, 45, 425, 638, 505, 26, 1409, 1448, 605, 232, 1459, 821, 1394, 1376, 809, 1163, 216, 791]
    )
    ds = FCMatrixDataset('../data/csv/severe_rds_v2.csv','../data/fetched/25751/raw', '25751', mrmr=mrmr_features, mapping=None)
    # ds2 = FCMatrixDataset('../data/csv/severe_rds_v2.csv','../data/fetched/raw/25751', '25751', None, mapping=None)

    print(ds[0][0][1140])
    print(ds2[0][0][0])


    # ds = FCMatrixDataset('data/gal_eids/gal_data.csv','data/fetched/25753_gal', '25753', None)
    # ds = FCMatrixDataset('../data/fully_balanced_ukb_filtered_25753_harir_mh_upto69.csv','../data/fetched/25751', '25751', 1)
    total_samples = len(ds)

    print(total_samples)
    train_ratio = 0.8
    test_ratio = 0.1
    val_ratio = 0.1

    # Ensure the training set is divisible by 4
    # Start with a direct calculation
    train_size = int(total_samples * train_ratio)
    # Adjust train_size to make it divisible by 4
    while train_size % 4 != 0:
        train_size -= 1

    # Calculate the remaining samples after allocating to the training set
    remaining_samples = total_samples - train_size
    # Assuming we want test and validation sets to be the same size
    test_size =  remaining_samples // 2
    val_size = remaining_samples - test_size

    # Ensure test and validation sets are of the same size and adjust if necessary
    # This block is to make sure we handle the division properly
    # In your case, since you prefer them to be equal and 10% each, adjustments might be minor
    if test_size + val_size + train_size < total_samples:
        extra = total_samples - (test_size + val_size + train_size)
        test_size += extra // 2
        val_size += extra // 2

    print(f"Train size: {train_size}, Test size: {test_size}, Validation size: {val_size}")

    train, val, test = balanced_random_split_v2(ds, [train_size, val_size, test_size], 4)

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
