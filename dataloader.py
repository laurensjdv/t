

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader



from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import SubsetRandomSampler



class FCMatrixDataset(Dataset):
    def __init__(self, ukb_filtered, dir, data_field, mapping):
        self.ukb_filtered = pd.read_csv(ukb_filtered, sep=' ')
        self.dir = dir
        self.data_field = data_field
        if mapping is not None:
            self.mapping ={'HC': 0, 'single ep.': 1, 'moderate rMDD': 2, 'severe rMDD': 3}
        else:
            self.mapping = [0, 1]



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
    # ds = FCMatrixDataset('data/ukb_filtered_25753_harir_mh_upto69.csv','data/fetched/25753', '25753')
    ds = FCMatrixDataset('data/gal_eids/gal_data.csv','data/fetched/25753_gal', '25753', None)
    total_size = len(ds)
    train_size = int(.8* total_size)
    val_size = int(.1 * total_size)
    test_size = total_size - train_size - val_size

    train, val, test = random_split(ds, [train_size, val_size, test_size])

    batch_size = 128

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    labels = []

    for matrix, label in train_loader:
        print(label)
        labels.extend(label)
        

    for matrix, label in test_loader:
        labels.extend(label)

    for matrix, label in val_loader:
        labels.extend(label)
    # matrix, label = next(iter(train_loader))
    print(matrix)

    label_counts = {0:0, 1:0, 2:0, 3:0}
    for label in labels:
        label_counts[label.item()] += 1

    print(label_counts)

    # matrix, label = ds[0]
    # D = 55
    # sq_fc = np.zeros((D,D))
    # sq_fc[np.triu_indices(D,1)] = matrix.split()
    # sq_fc += sq_fc.T

    # print(sq_fc.shape)

    # plt.imshow(sq_fc)
    # plt.colorbar()
    # plt.show()