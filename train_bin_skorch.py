from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from models.mlp import MLP
from models.bin_mlp import binMLP
from dataloader import FCMatrixDataset


from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from skorch import NeuralNetClassifier

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import SubsetRandomSampler


import torch
import torch.nn as nn
import torch.optim as optim

from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataset, lr, batch_size, epochs, seed, data_dir):
    """
    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    """


    np.random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():  # GPU seed
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.determinstic = True
    #     torch.backends.cudnn.benchmark = False

    ds = FCMatrixDataset(dataset, data_dir, '25753', None)

    total_size = len(ds)
    train_size = int(.8* total_size)
    val_size = int(.1 * total_size)
    test_size = total_size - train_size - val_size

    train, val, test = random_split(ds, [train_size, val_size, test_size])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)


    model = NeuralNetClassifier(
        module=binMLP,
        module__n_inputs=1485,
        criterion=nn.BCELoss,
        # optimizer=optim.Adam,
        max_epochs=100,,
        # optimizer__weight_decay=0.0001,

    )
    
    # define the grid search parameters
    param_grid = {
        'optimizer': [optim.Adam, optim.AdamW],
        # 'batch_size': [512, 128, 64, 32],
        'batch_size': [64],
        'max_epochs': [50, 100],
        # 'lr': [0.0001, 0.00005, 0.00001],
        'lr': [0.00005],

        'optimizer__weight_decay': [0.01, 0.001, 0.0001],
        'module__dropout': [0.0, 0.1, 0.2, 0.5],
        'module__n_hidden': [[512], [512,256,64], [512,256,256,128,64,32,16]]

    }
    gs = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, refit=False, scoring='accuracy', verbose=0, cv=2)
    for i, (inputs, targets) in enumerate(train_loader):
        inputs#.to(DEVICE)
        ros = RandomOverSampler(random_state=seed)
        targets = targets.unsqueeze(1).float()#.to(DEVICE)

        grid_result = gs.fit(inputs, targets)

    print('SEARCH COMPLETE')
    print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))

    # with open(f"outputs/grid_search/best_param.yml", 'w') as f:
    #     f.write(f"{str(gs.best_score_)}\n")
    #     f.write(f"{str(gs.best_params_)}\n")

    return gs.best_estimator_

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--dataset",
        default= 'data/gal_eids/gal_data.csv',
        type=str,
        nargs="+",
        help='Path to dataset contaning eids and labels. Example: "data/gal_eids/gal_data.csv"',
    )



    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=512, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=100, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/fetched/25753_gal",
        type=str,
        help="Data directory where to find the dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    model = train(**kwargs)


