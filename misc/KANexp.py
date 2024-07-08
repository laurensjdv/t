from kan import *
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from dataloaders.dataloader import FCMatrixDataset
from utils import balanced_random_split_v2

from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"


batch_size = 48
n_epochs = 25
hidden_dims = [256,64]
lr = 0.00007697
mrmr = False
lamb = 0.1
lamb_entropy = 2.0
grid_size = 5

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=99)

if mrmr is True:
	input_features = 55
	mrmr_features = np.array([1140, 536, 223, 907, 1449, 499, 1293, 45, 135, 1440, 879, 1384, 1210, 1316, 122, 22, 492, 638, 765, 1027, 1464, 501, 1462, 395, 26, 1079, 70, 425, 1403, 1409, 1318, 886, 1459, 1448, 939, 1163, 547, 10, 413, 676, 131, 216, 942, 1136, 1386, 232, 1455, 1337, 814, 139, 392, 1376, 1382, 471, 656]
	)
# smith moderate features
# mrmr_features = np.array([922, 135, 956, 867, 589, 1308, 55, 426, 1483, 466, 1023, 1324, 1327, 902, 1391, 981, 200, 275, 871, 1122, 1314, 590, 934, 629, 952, 421, 538, 762, 378, 1053, 865, 418, 948, 462, 248, 646, 944, 485, 539, 1269, 407, 879, 1027, 782, 1449, 1421, 1360, 550, 1257, 415, 681, 1147, 89, 664, 163]
#                          )
else:
	input_features = 1485
	mrmr_features = None

dims = [input_features] + hidden_dims + [2]

ds = "data/csv/severe_rds.csv"
# ds = "data/csv/balanced_sex_classification.csv"
data_dir = "data/fetched/25751/raw"
udi = "25751"

dataset = FCMatrixDataset(ds, data_dir, udi, mapping=None, mrmr=mrmr_features)
labels = np.genfromtxt(ds)
eids = labels[:,0]
labels = labels[:,1]
eval_metrics = np.zeros((skf.n_splits, 3))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=99)

def KAN_train(model, loader):
    # model.train()

    loss_all = 0
    for data in loader:
        inputs = data[0].to(DEVICE)
        labels = data[1].to(DEVICE)

        optimizer.zero_grad()
        output = model(inputs)
       
        loss = func.cross_entropy(output, labels)
        loss.backward()
        loss_all += batch_size * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def KAN_test(model, loader):
    pred = []
    label = []
    loss_all = 0
    for data in loader:
        inputs = data[0].to(DEVICE)
        labels = data[1].to(DEVICE)

        output = model(inputs)
        loss = func.cross_entropy(output, labels)
        loss_all += batch_size * loss.item()
        # pred.append(func.softmax(output, dim=1).max(dim=1)[1])
        _, predicted = torch.max(output, 1)
        pred.append(predicted)
        label.append(labels)

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    print(y_pred)
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()
    print(y_true)
    tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
    
    epoch_sen = tp / (tp + fn)
    # epoch_sen = 0
    epoch_spe = tn / (tn + fp)
    # epoch_spe = 0
    epoch_acc = (tn + tp) / (tn + tp + fn + fp)
    return epoch_sen, epoch_spe, epoch_acc, loss_all / len(val_dataset)

print(DEVICE)

for n_fold, (train_val, test) in enumerate(skf.split(labels, labels)):

    print(dims)

    model = KAN(dims, symbolic_enabled=False, device=DEVICE, bias_trainable=True, grid=grid_size)



    train_val_dataset, test_dataset = Subset(dataset, train_val), Subset(dataset, test)
    train_val_labels = labels[train_val]
    train_val_index = np.arange(len(train_val_dataset))

    train, val, _, _ = train_test_split(train_val_index, train_val_labels, test_size=0.11, shuffle=True, stratify=train_val_labels)
    train_dataset, val_dataset = Subset(train_val_dataset, train), Subset(train_val_dataset, val)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def seperate_subset(subset):
        input = [subset[i][0] for i in range(len(subset))]
        input = torch.stack(input)
        label = [subset[i][1] for i in range(len(subset))]
        
        label = torch.stack(label)
        return input, label

    train_input, train_label = seperate_subset(train_dataset)
    val_input, val_label = seperate_subset(val_dataset)
    test_input, test_label = seperate_subset(test_dataset)

    # lr=.001

    datadict = {'train_input': train_input, 'train_label': train_label, 'val_input': val_input, 'val_label': val_label, 'test_input': test_input, 'test_label': test_label}
    results = model.train(datadict, opt="LBFGS", loss_fn=torch.nn.CrossEntropyLoss(), steps=n_epochs, lr=lr, lamb=lamb, lamb_entropy=lamb_entropy, update_grid=True, device=DEVICE)
    print(results['train_loss'])
    
    # print('CV: {:03d}, Epoch: {:03d}, Val Loss: {:.5f}, Val BAC: {:.5f}'.format(n_fold + 1, epoch + 1, min_v_loss, best_val_acc))
    
    test_sen, test_spe, test_acc, _ = KAN_test(model, test_loader)
    print('CV: {:03d}, SEN: {:.5f}, SPE: {:.5f}, ACC: {:.5f}'.format(n_fold + 1, test_sen, test_spe, test_acc))


    eval_metrics[n_fold, 0] = test_sen
    eval_metrics[n_fold, 1] = test_spe
    eval_metrics[n_fold, 2] = test_acc

eval_df = pd.DataFrame(eval_metrics)
eval_df.columns = ['SEN', 'SPE', 'ACC']
eval_df.index = ['Fold_%02i' % (i + 1) for i in range(skf.n_splits)]
print(eval_df)
print('Average Sensitivity: %.4f±%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
print('Average Specificity: %.4f±%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
print('Average Accuracy: %.4f±%.4f' % (eval_metrics[:, 2].mean(), eval_metrics[:, 2].std()))
