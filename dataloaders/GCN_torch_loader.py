import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from utils import balanced_random_split_v2, compute_KNN_graph, fc_to_matrix, adjacency

from torch_geometric.utils import dense_to_sparse



class GraphDataset(Dataset):
    def __init__(self, ukb_filtered, dir, data_field, mapping, oversample=False):
        self.ukb_filtered = pd.read_csv(ukb_filtered, sep=' ', header=None)
        self.dir = dir
        self.data_field = data_field
        self.mapping = mapping

        self.raw_paths = self.raw_file_names()
        self.processed_data = self.process()

    
    def raw_file_names(self):
        l = []
        for i in range(len(self.ukb_filtered)):
            eid = self.ukb_filtered.iloc[i, 0]
            filename = str(eid) + '_' + self.data_field + '_2_0.txt'
            l.append(filename)
        return l
    
    def process(self, k_degree=10):
        data_list = []
        i = 0
        for raw_path in self.raw_paths:
            matrix_path = os.path.join(self.dir, raw_path)
            label = self.ukb_filtered.iloc[i, 1]  

            if self.mapping is not None:
                enc_label = self.mapping[label]
            else:
                enc_label = label
            with open(matrix_path, "r") as f:
                matrix = f.read().split()
            matrix = [float(item) for item in matrix]
            D = int((1 + np.sqrt(1 + 8 * len(matrix))) / 2)
            matrix = fc_to_matrix(matrix, D)
            x = torch.FloatTensor(matrix)
            adj = compute_KNN_graph(matrix, k_degree=k_degree)
            adj = torch.from_numpy(adj).float()
            edge_index, edge_attr = dense_to_sparse(adj)
            data_list.append([x, edge_index, edge_attr, enc_label])
            i += 1
        return data_list
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        data = self.processed_data[idx]
        x = data[0]
        edge_index = data[1]
        edge_attr = data[2]
        y = data[3]
        return x, edge_index, edge_attr, y
        

