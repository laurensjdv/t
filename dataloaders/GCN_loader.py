

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pathlib import Path


import sys
import os

# Assuming your current script is directly inside a subfolder of 't'
# Get the absolute path to the 't' folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Check if the parent directory's name is 't'. If not, adjust the path accordingly.
if os.path.basename(parent_dir) == 't':
    sys.path.append(parent_dir)
else:
    raise Exception("The script is not located in a subfolder of 't'")

# Import the utils module from the 't' folder

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

import torch
from torch.utils.data import  Subset

from imblearn.over_sampling import RandomOverSampler

from torch.utils.data.dataset import random_split
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Dataset, Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data.collate import collate
from scipy.sparse import coo_matrix

from collections import Counter

from torch.utils.data import SubsetRandomSampler

from utils import balanced_random_split_v2, compute_KNN_graph, fc_to_matrix, adjacency
from utils import balanced_random_split


class GraphDataset(InMemoryDataset):
    def __init__(self, ukb_filtered, root, data_field, oversample=False, transform=None, pre_transform=None, pre_filter=None, mapping=None, mrmr=None):
        
        print(ukb_filtered)
        self.ukb_filtered = pd.read_csv(ukb_filtered, sep=' ', header=None)
        self.data_field = data_field
        self.mapping = mapping
        self.mrmr = mrmr
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        print(ukb_filtered)
        
        if oversample:
            ros = RandomOverSampler(random_state=0)

    @property
    def raw_file_names(self):
        l = []
        # ukb_filtered = pd.read_csv(self.ukb_filtered, sep=' ', header=None)
        for i in range(len(self.ukb_filtered)):
            eid = self.ukb_filtered.iloc[i, 0]
            filename = str(eid) + '_' + self.data_field + '_2_0.txt'
            l.append(filename)
        return l
    
    @property
    def processed_file_names(self):
        return 'data.pt'


    def process(self, k_degree=10):
        data_list = []
        i = 0
        for raw_path in self.raw_paths:
            matrix_path = raw_path
            label = self.ukb_filtered.iloc[i, 1]  

            if self.mapping is not None:
                enc_label = self.mapping[label]
            else:
                enc_label = label
            with open(matrix_path, "r") as f:
                matrix = f.read().split()
            matrix = np.array([float(item) for item in matrix])
            if self.mrmr is not None:
                matrix = matrix[self.mrmr]
            D = int((1 + np.sqrt(1 + 8 * len(matrix))) / 2)
            matrix = fc_to_matrix(matrix, D)
            x = torch.FloatTensor(matrix)
            adj = compute_KNN_graph(matrix, k_degree=k_degree)
            adj = torch.from_numpy(adj).float()
            edge_index, edge_attr = dense_to_sparse(adj)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data.y = torch.tensor([enc_label])

            data_list.append(data)
            i += 1
        self.save(data_list, self.processed_paths[0])



if __name__ == "__main__":
    # mapping = {'HC': 0, 'single ep.': 1, 'moderate rMDD': 1, 'severe rMDD': 1}
    # mapping = {'HC': 0, 'severe rMDD': 1}
    mapping = {'HC': 0, 'MDD': 1}
    # ds = GraphDataset('../data/ukb_filtered_25753_harir_mh_upto69.csv','../data/fetched/25751', '25751', mapping)
    # ds = FCMatrixDataset('data/gal_eids/gal_data.csv','data/fetched/25753_gal', '25753', None)
    # ds = GraphDataset('../data/fully_single_balanced_ukb_filtered_25753_harir_mh_upto69.csv','../data/fetched/25751', '25751', mapping)
    ds = GraphDataset('../data/2_class_balanced_ukb_filtered_25753_harir_mh_upto69.csv','../data/fetched/25753', '25753', mapping, oversample=True)

    # print class of ds
    print(ds.__class__)
    print(ds.__class__.__name__)
    # print(f"y: {ds[0][1]}")
    # print(ds[0][0].edge_attr)
    # ds.process(k_degree = 1)
    # print(ds[0][0].edge_attr)
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

    train, val, test = balanced_random_split_v2(ds, [train_size, val_size, test_size], 2)
    # train, val, test = balanced_random_split(ds, [train_size, val_size, test_size])
    # train, val, test = random_split(ds, [train_size, val_size, test_size])
    batch_size = 128

 

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)


    all_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    train_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    val_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    test_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    print("train loop")
    for i, batch in enumerate(train_loader):
        input = batch.x
        targets = batch.y
        # edge_index = input.edge_index
        # edge_attr = input.edge_attr
        # print(x.shape)
        # print(edge_index.shape)
        # print(edge_attr.shape)
        # exit()
        for t in targets:
            train_counts[t.item()] += 1
            all_counts[t.item()] += 1
    
    print("val loop")
    for i, batch in enumerate(val_loader):
        input = batch.x
        targets = batch.y
        for t in targets:
            val_counts[t.item()] += 1
            all_counts[t.item()] += 1
    print("test loop")
    for i, batch in enumerate(test_loader):
        input = batch.x
        targets = batch.y
        for t in targets:
            test_counts[t.item()] += 1
            all_counts[t.item()] += 1

    print(all_counts)
    print(train_counts)
    print(val_counts)
    print(test_counts)

