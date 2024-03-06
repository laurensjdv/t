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
from dataloader import balanced_random_split

import wandb

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

from sklearn.linear_model import ElasticNet


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

def evaluate_model(model, data_loader, num_classes=2):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.
    """
    model.eval()
    loss_module = nn.BCELoss()
    losses = []

    all_preds = []
    all_targets = []

    for batch in data_loader:
        inputs = batch[0].to(DEVICE)
        targets = batch[1].unsqueeze(1).float().to(DEVICE)
            # targets = targets.unsqueeze(1).float().to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_module(outputs, targets)
            pred = torch.round(outputs).detach()


        losses.append(loss.item())
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
    cm = confusion_matrix(all_targets, all_preds, labels = range(num_classes))
    # precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, labels=range(num_classes), average=None)

    accuracy = np.trace(cm) / np.sum(cm)
    metrics = {
        "loss": np.mean(losses),
        "accuracy": accuracy,
        # "precision": precision,
        # "recall": recall,
        # "f1": f1,
        "conf_mat": cm
    }

    return metrics, np.mean(losses)

def train(config, dataloaders):

    config = wandb.config


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--dataset",
        default="data/gal_eids/gal_data.csv",
        type=str,
        nargs="+",
        help='Path to dataset contaning eids and labels. Example: "data/gal_eids/gal_data.csv"',
    )
    # parser.add_argument(
    #     "--model_type",
    #     default="bin_mlp",
    #     type=str,
    #     help='Model to use. Example: "bin_mlp" or "elasticnet',
    # )

    # Optimizer hyperparameters
    # parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate to use")
    # parser.add_argument("--batch_size", default=512, type=int, help="Minibatch size")

    # Other hyperparameters
    # parser.add_argument("--epochs", default=100, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/fetched/25751_gal",
        type=str,
        help="Data directory where to find the dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    # sweep_config = {
    # 'method': 'random'
    # }

    # sweep_config['metric'] = {
    # 'name': 'loss',
    # 'goal': 'minimize'   
    # }

    # sweep_config['metric'] = {
    #     'name': 'accuracy',
    #     'goal': 'maximize'
    # }

    # parameters_dict = {
    # 'optimizer': {
    #     'values': ['adam', 'sgd']
    #     },
    # 'fc_layer_size': {
    #     'values': [128, 256, 512]
    #     },
    # 'dropout': {
    #       'distribution': 'uniform',
    #       'min': 0.0,
    #       'max': 1.0
    #     },
    # 'learning_rate': {
    #     'distribution': 'uniform',
    #     'min': 1e-4,
    #     'max': 1e-6
    #   },
    # 'batch_size': {
    #     # integers between 32 and 256
    #     # with evenly-distributed logarithms 
    #     'distribution': 'q_log_uniform_values',
    #     'q': 8,
    #     'min': 32,
    #     'max': 512
    #   }
    # }

    sweep_config = 'elasticnet_config.yaml'

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    # sweep_config['parameters'] = parameters_dict


    wandb.agent(sweep_id, train, count=5)

    


    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=100, timeout=600)

    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))

    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value: ", trial.value)

    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))
