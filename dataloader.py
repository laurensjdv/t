import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader


class FCMatrixDataset(Dataset):
    def __init__(self, ukb_filtered, dir, data_field):
        self.ukb_filtered = pd.read_csv(ukb_filtered, sep=' ')
        self.dir = dir
        self.data_field = data_field


    def __len__(self):
        return len(self.ukb_filtered)

    def __getitem__(self, idx):
        eid = self.ukb_filtered.iloc[idx, 0]
        filename = str(eid) + '_' + self.data_field + '_2_0.txt'
        matrix_path = os.path.join(self.dir, filename) 
        label = self.ukb_filtered.iloc[idx, 1]
        f = open(matrix_path, "r")
        matrix = f.read()

        return matrix, label

ds = FCMatrixDataset('data/ukb_filtered_25753_harir_mh_upto69.csv','data/fetched/25753', '25753')

matrix, label = ds[0]
D = 55
sq_fc = np.zeros((D,D))
sq_fc[np.triu_indices(D,1)] = matrix.split()
sq_fc += sq_fc.T

print(sq_fc.shape)

plt.imshow(sq_fc)
plt.colorbar()
plt.show()