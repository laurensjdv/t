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


def load_data(data_dir, dataset):
    udi = data_dir.split("/")[-1]
    udi = udi.split("_")[0]
    ds = FCMatrixDataset(data_dir, dataset, udi, None)

    total_size = len(ds)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train, val, test = balanced_random_split(ds, [train_size, val_size, test_size])

    return train, val, test


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

    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    # precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, labels=range(num_classes), average=None)

    accuracy = np.trace(cm) / np.sum(cm)
    metrics = {
        "loss": np.mean(losses),
        "accuracy": accuracy,
        # "precision": precision,
        # "recall": recall,
        # "f1": f1,
        "conf_mat": cm,
    }

    return metrics, np.mean(losses)


def train(config):
    with wandb.init(config=config):
        config = wandb.config

        epochs = config.epochs

        # Set seed

        train_loader = DataLoader(
            config["train"], batch_size=config["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            config["val"], batch_size=config["batch_size"], shuffle=True
        )
        test_loader = DataLoader(
            config["test"], batch_size=config["batch_size"], shuffle=True
        )

        model = binMLP(n_inputs=1485, n_hidden=config.hidden_dims).to(DEVICE)

        loss_module = nn.BCELoss()
        if config["optimizer"] == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
        else:
            optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

        train_loss = np.zeros(epochs)
        val_loss = np.zeros(epochs)
        train_acc = np.zeros(epochs)
        val_acc = np.zeros(epochs)
        best_val_acc = 0

        for epoch in range(config["epochs"]):
            model.train()

            train_losses = []
            all_preds = []
            all_targets = []
            for inputs, targets in train_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.unsqueeze(1).float().to(DEVICE)

                outputs = model(inputs)
                loss = loss_module(outputs, targets)
                loss += model.l1_reg()
                loss += model.l2_reg()

                train_losses.append(float(loss))

                pred = torch.round(outputs).detach()

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss[epoch] = np.mean(train_losses)
            train_metrics, _ = evaluate_model(model, train_loader)
            train_acc[epoch] = train_metrics["accuracy"]

            val_metrics, val_loss[epoch] = evaluate_model(model, val_loader)
            val_acc[epoch] = val_metrics["accuracy"]
            val_loss[epoch] = val_metrics["loss"]

            if val_acc[epoch] > best_val_acc:
                best_val_acc = val_acc[epoch]
                best_model = deepcopy(model)

            wandb.log(
                {
                    "train_loss": train_loss[epoch],
                    "val_loss": val_loss[epoch],
                    "train_acc": train_acc[epoch],
                    "val_acc": val_acc[epoch],
                    "epoch": epoch,
                }
            )

        # test best model
        test_metrics, _ = evaluate_model(best_model, test_loader)
        test_acc = test_metrics["accuracy"]

        wandb.log({"test_acc": test_acc})


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

    train_d, val_d, test_d = load_data(**kwargs)

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

    sweep_config = "elasticnet_config.yaml"

    # add data to config

    sweep_config["train"] = train_d
    sweep_config["val"] = val_d
    sweep_config["test"] = test_d

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    # sweep_config['parameters'] = parameters_dict

    wandb.agent(sweep_id, train, count=50)

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
