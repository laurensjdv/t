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

from imblearn.over_sampling import RandomOverSampler


from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import SubsetRandomSampler


import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        targets = batch[1].to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_module(outputs, targets)
            _, pred = torch.max(outputs, 1)

        losses.append(loss.item())
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))


    accuracy = np.trace(cm) / np.sum(cm)
    metrics = {
        "loss": np.mean(losses),
        "accuracy": accuracy,
        "conf_mat": cm,
    }

    return metrics, np.mean(losses)


def train(dataset, hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir, oversampling=False):
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

    ds = FCMatrixDataset(dataset, data_dir, "25753", 1)

    total_size = len(ds)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train, val, test = random_split(ds, [train_size, val_size, test_size])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    if oversampling:
        train_X = torch.tensor([batch[0] for batch in train_loader]).to(DEVICE)
        train_y = torch.tensor([batch[1] for batch in train_loader]).to(DEVICE)

        train_Y_counts = np.bincount(train_y)
        train_Y_max = train_Y_counts.max()
        sampling_strategy = {i: train_Y_max for i in range(4)}
        ros = RandomOverSampler(random_state=seed, sampling_strategy=sampling_strategy)

        train_X_resampled, train_y_resampled = ros.fit_resample(train_X, train_y)
        
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(train_X_resampled, train_y_resampled), batch_size=batch_size, shuffle=True
        )
    # Initialize model and loss module
    model = MLP(1485, hidden_dims, 4).to(DEVICE)
    # model = binMLP(1485, hidden_dims).to(DEVICE)
    print(model)

    loss_module = nn.CrossEntropyLoss()
    # loss_module = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
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
        for inputs, targets in tqdm(train_loader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            outputs = model(inputs)
            loss = loss_module.forward(outputs, targets)
            train_losses.append(float(loss))
            _, pred = torch.max(outputs, 1)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss[epoch] = np.mean(train_losses)
        cm = confusion_matrix(all_targets, all_preds, labels=range(4))
        accuracy = np.trace(cm) / np.sum(cm)
        train_accuracies[epoch] = accuracy

        val_metrics, val_loss = evaluate_model(model, val_loader)
        val_losses[epoch] = val_loss
        val_accuracies[epoch] = val_metrics["accuracy"]

        print(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_model = deepcopy(model)
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

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--dataset",
        default="data/ukb_filtered_25753_harir_mh_upto69.csv",
        type=str,
        nargs="+",
        help='Path to dataset contaning eids and labels. Example: "data/gal_eids/gal_data.csv"',
    )
    parser.add_argument(
        "--hidden_dims",
        default=[512, 512, 128, 128, 64, 32],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"',
    )
    parser.add_argument(
        "--use_batch_norm",
        action="store_true",
        help="Use this option to add Batch Normalization layers to the MLP.",
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/fetched/25753",
        type=str,
        help="Data directory where to find the dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_info = train(**kwargs)

    print("Test accuracy: ", test_accuracy)
    # print("f1 score: ", logging_info["test_metrics"]["f1_beta"])

    train_loss = logging_info["train_loss"]
    val_losses = logging_info["val_loss"]
    test_metrics = logging_info["test_metrics"]

    plt.plot(train_loss, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.show()

    plt.plot(val_accuracies, label="val_acc")
    plt.plot(logging_info["train_acc"], label="train_acc")
    plt.legend()

    plt.matshow(test_metrics["conf_mat"])

    # add legend to confusion matrix
    plt.colorbar()
    # add axis labels to confusion matrix
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.show()

    print(val_accuracies)
    print(test_accuracy)
