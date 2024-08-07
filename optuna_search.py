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
from t.dataloaders.dataloader import FCMatrixDataset
from t.dataloaders.dataloader import balanced_random_split

import optuna
from optuna.trial import TrialState


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


def objective(trial):
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

    
    dataset = "data/gal_eids/gal_data.csv"
    seed = 100
    data_dir = "data/fetched/25753_gal"
    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the dataset
    # cifar10 = cifar10_utils.get_cifar10(data_dir)
    # cifar10_loader = cifar10_utils.get_dataloader(
    #     cifar10, batch_size=batch_size, return_numpy=False
    # )

    ds = FCMatrixDataset(dataset, data_dir, "25753", None)

    total_size = len(ds)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train, val, test = balanced_random_split(ds, [train_size, val_size, test_size])

    # hidden_dims = trial.suggest_categorical(
    #     "hidden_dims", [[512], [512, 256, 64], [512, 256, 256, 128, 64, 32, 16]]
    # )
    hidden_dims = [512]
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 512, log=True)
    epochs = trial.suggest_categorical("epochs", [50, 100])
    l1_lambda = trial.suggest_float("l1_lambda", 0.0, 1.0, log=False)
    l2_lambda = 1 - l1_lambda

    # l1_lambda = 0
    # l2_lambda = 0
    dropout = trial.suggest_float("dropout", 0.1, 0.9, log=True)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    test = []
    for matrix, label in val_loader:
        test.extend(label)

    test_counts = {0:0, 1:0, 2:0, 3:0}
    for label in test:
        test_counts[label.item()] += 1

    print(test_counts)

    # Initialize model and loss module
    # model = MLP(1485, hidden_dims, 4).to(DEVICE)
    model = binMLP(1485, hidden_dims, dropout).to(DEVICE)

   

    # loss_module = nn.CrossEntropyLoss()
    loss_module = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # optimizer= optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    # Training loop including validation
    train_loss = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    train_accuracies = np.zeros(epochs)
    val_accuracies = np.zeros(epochs)
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()

        train_losses = []
        all_preds = []
        all_targets = []
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.unsqueeze(1).float().to(DEVICE)

            # print device of input and targets

            # Forward pass
            outputs = model(inputs)
            loss = loss_module.forward(outputs, targets)

            loss = loss + model.l1_loss(l1_lambda) + model.l2_loss(l2_lambda)
            train_losses.append(float(loss))
            pred = torch.round(outputs).detach()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss[epoch] = np.mean(train_losses)
        cm = confusion_matrix(all_targets, all_preds, labels=range(2))
        accuracy = np.trace(cm) / np.sum(cm)
        train_accuracies[epoch] = accuracy
        val_metrics, val_loss = evaluate_model(model, val_loader)
        val_losses[epoch] = val_loss
        val_accuracies[epoch] = val_metrics["accuracy"]

        # print(val_metrics["accuracy"])
        trial.report(val_metrics["accuracy"], epoch)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_model = deepcopy(model)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Test best model
    test_metrics, test_loss = evaluate_model(best_model, test_loader)
    test_accuracy = test_metrics["accuracy"]
    # Add any information you might want to save for plotting
    logging_info = {
        "train_loss": train_loss,
        "val_loss": val_losses,
        "train_acc": train_accuracies,
        "val_acc": val_accuracies,
        "test_metrics": test_metrics,
    }

    return test_accuracy


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

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
