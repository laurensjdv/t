

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from pathlib import Path


import torch
from torch.utils.data import Dataset, DataLoader, Subset

from imblearn.over_sampling import RandomOverSampler

from torch.utils.data.dataset import random_split
from utils import balanced_random_split_v2

from collections import Counter
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import SubsetRandomSampler



class FCMatrixDataset(InMemoryDataset):
    def __init__(self, ukb_filtered, root, data_field, oversample=False, transform=None, pre_transform=None, pre_filter=None):
        
        self.ukb_filtered = ukb_filtered
        self.data_field = data_field

        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        print(ukb_filtered)
        
        if oversample:
            ros = RandomOverSampler(random_state=0)

    # @property
    # def raw_file_names(self):
    #     l = []
    #     # ukb_filtered = pd.read_csv(self.ukb_filtered, sep=' ', header=None)
    #     for i in range(len(self.ukb_filtered)):
    #         eid = self.ukb_filtered.iloc[i, 0]
    #         filename = str(eid) + '_' + self.data_field + '_2_0.txt'
    #         l.append(filename)
    #     return l
    @property
    def raw_file_names(self):
        file_paths = sorted(list(Path(self.raw_dir).glob('*.txt')))
        return [str(file_path.name) for file_path in file_paths]
    @property
    def processed_file_names(self):
        return 'data.pt'


    def process(self):
        labels = np.genfromtxt(self.ukb_filtered)
        labels = labels[:,1]
        
        data_list = []
        i = 0
        for raw_path, label  in zip(self.raw_paths, labels):
            with open(raw_path, "r") as f:
                matrix = f.read().split()
            matrix = [float(item) for item in matrix]
            x = torch.FloatTensor([matrix])

            
            data = Data(x=x, y = torch.tensor([label]).long())
            data_list.append(data)
            i += 1
        self.save(data_list, self.processed_paths[0])



    # def __getitem__(self, idx):
    #     eid = self.ukb_filtered[:,0]
    #     print(type(eid))
    #     print(idx)
    #     eid = eid[idx]
    #     print(type(eid))        
    #     filename = str(eid) + '_' + self.data_field + '_2_0.txt'

    #     matrix_path = os.path.join(self.dir, filename)
    #     label = self.ukb_filtered[:,1][idx]
    #     print(label)
    #     if self.mapping is not None:
    #         # label HC, single ep., moderate rMDD, severe rMDD
    #         enc_label = self.mapping[label]
    #     else:
    #         enc_label = label
        

    #     with open(matrix_path, "r") as f:
    #         matrix = f.read().split()
    #     matrix = [float(item) for item in matrix]
    #     matrix = torch.FloatTensor(matrix)
    #     return matrix, enc_label

if __name__ == "__main__":
    ds = FCMatrixDataset('../data/ukb_filtered_25753_harir_mh_upto69.csv','../data/fetched/25753', '25753', 1)
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
