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

from functools import partial

import wandb
import yaml

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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")


def evaluate_model(model, data_loader, num_classes=4):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.
    """
    model.eval()
    loss_module = nn.CrossEntropyLoss()
    losses = []

    all_preds = []
    all_targets = []

    for batch in data_loader:
        inputs = batch[0].to(DEVICE)
        # targets = batch[1].unsqueeze(1).float().to(DEVICE)
        targets = batch[1].to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_module(outputs, targets)
            _, pred = torch.max(outputs, 1)


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


def train(dataset, seed, data_dir, udi, config=None, oversampling=False):


    ds = FCMatrixDataset('data/fully_balanced_ukb_filtered_25753_harir_mh_upto69.csv', data_dir, udi, 1)

    total_size = len(ds)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_d, val_d, test_d = random_split(ds, [train_size, val_size, test_size])

    # set seed 

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False


    with wandb.init(config=config):
        config = wandb.config
        epochs = config.epochs
        batch_size = config.batch_size

        train_loader = DataLoader(
            train_d, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_d, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_d, batch_size=batch_size, shuffle=True
        )
        wandb.log({"oversampling": oversampling})
        if oversampling:
            train_X = torch.tensor([])
            train_Y = torch.tensor([], dtype=torch.int64)
            for batch in train_loader:
                inputs = batch[0]
                targets = batch[1]
                train_X = torch.cat((train_X, inputs), 0)
                train_Y = torch.cat((train_Y, targets), 0)

            train_Y_counts = np.bincount(train_Y)
            max_idx = np.argmax(train_Y_counts)
            train_Y_2nd_max = np.partition(train_Y_counts, -2)[-2]
            sampling_strategy = {i: train_Y_2nd_max for i in range(4)}
            sampling_strategy[max_idx] = train_Y_counts[max_idx]

            ros = RandomOverSampler(random_state=seed, sampling_strategy=sampling_strategy)

            train_X_resampled, train_Y_resampled = ros.fit_resample(train_X, train_Y)

            train_Y_resampled = torch.tensor(train_Y_resampled, dtype=torch.int64)
            train_X_resampled = torch.tensor(train_X_resampled, dtype=torch.float32)

            train_data = torch.utils.data.TensorDataset(train_X_resampled, train_Y_resampled)
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True
            )
        if udi == "25755":
            model = MLP(55, config.hidden_dims, 4).to(DEVICE)
        else:
            model = MLP(1485, config.hidden_dims, 4).to(DEVICE)


        loss_module = nn.CrossEntropyLoss()
        if config["optimizer"] == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
            l1_lambda = 0
            l2_lambda = 0
        else:
            optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
            l1_lambda = config["l1_lambda"]
            l2_lambda = 1 - l1_lambda

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
                targets = targets.to(DEVICE)
                outputs = model(inputs)
                loss = loss_module(outputs, targets)

                loss += model.l1_loss(l1_lambda)
                loss += model.l2_loss(l2_lambda)
                train_losses.append(float(loss))

                _ , pred = torch.max(outputs, 1)

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

    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/fetched/25753",
        type=str,
        help="Data directory where to find the dataset.",
    )
    parser.add_argument(
        "--sweep_config",
        default="elasticnet_config.yaml",
        type=str,
        help="Path to the sweep configuration file.",
    )
    parser.add_argument(
        "--oversampling",
        action="store_true",
        help="Use this option to add oversampling to the dataset.",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    data_dir = kwargs["data_dir"]

    udi = data_dir.split("/")[-1]
    udi = udi.split("_")[0]

    sweep_config_path = kwargs["sweep_config"]

    with open(sweep_config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    train_partial = partial(train, seed=kwargs["seed"], dataset=kwargs["dataset"], data_dir=kwargs["data_dir"], udi=udi, oversampling=kwargs["oversampling"])

    print(sweep_config)

    
    sweep_id = wandb.sweep(sweep_config, project=f"smith_{kwargs['oversampling']}_{sweep_config['name']}_{udi}")

    wandb.agent(sweep_id, train_partial, count=100)

