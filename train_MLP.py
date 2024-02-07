from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

# import confusionmatrix from torchmetrics
from torchmetrics import ConfusionMatrix

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """
    conf_mat = torch.zeros((predictions.shape[1], predictions.shape[1]))
    for i in range(predictions.shape[0]):
        conf_mat[targets[i], torch.argmax(predictions[i])] += 1
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.0):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    metrics = {}
    metrics["accuracy"] = (
        torch.trace(confusion_matrix) / torch.sum(confusion_matrix)
    ).item()
    metrics["precision"] = torch.diag(confusion_matrix) / torch.sum(
        confusion_matrix, dim=0
    )
    metrics["recall"] = torch.diag(confusion_matrix) / torch.sum(
        confusion_matrix, dim=1
    )

    metrics["f1_beta"] = (
        (1 + beta**2)
        * metrics["precision"]
        * metrics["recall"]
        / (beta**2 * metrics["precision"] + metrics["recall"])
    )
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
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
    conf_mat = torch.zeros((num_classes, num_classes))
    conf_matpt = torch.zeros((num_classes, num_classes))
    losses = []

    for batch in data_loader:
        inputs = batch[0].reshape(batch[0].shape[0], -1).to(DEVICE)
        targets = batch[1].to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_module(outputs, targets)

        losses.append(loss.item())
        conf_mat += confusion_matrix(outputs, targets)
        mat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        conf_matpt += mat(outputs, targets)

    metrics = confusion_matrix_to_metrics(conf_mat)
    metrics["loss"] = np.mean(losses)
    metrics["conf_mat"] = conf_mat
    return metrics, np.mean(losses)


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

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
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=False
    )

    # Initialize model and loss module
    model = MLP(32 * 32 * 3, hidden_dims, 10, use_batch_norm=False).to(DEVICE)
    print(model)
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # Training loop including validation
    train_loss = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    train_accuracies = np.zeros(epochs)
    val_accuracies = np.zeros(epochs)
    best_val_acc = 0

    for epoch in tqdm(range(epochs)):
        model.train()

        conf_mat = torch.zeros((10, 10))
        train_losses = []
        for batch in cifar10_loader["train"]:
            inputs, targets = batch
            inputs = inputs.reshape(inputs.shape[0], -1).to(DEVICE)
            targets = targets.to(DEVICE)

            # print device of input and targets

            # Forward pass
            outputs = model(inputs)
            loss = loss_module.forward(outputs, targets)
            train_losses.append(float(loss))
            conf_mat += confusion_matrix(outputs, targets)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss[epoch] = np.mean(train_losses)
        train_metrics = confusion_matrix_to_metrics(conf_mat)
        train_accuracies[epoch] = train_metrics["accuracy"]

        val_metrics, val_loss = evaluate_model(model, cifar10_loader["validation"])
        val_losses[epoch] = val_loss
        val_accuracies[epoch] = val_metrics["accuracy"]

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_model = deepcopy(model)
    # Test best model
    test_metrics, test_loss = evaluate_model(best_model, cifar10_loader["test"])
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
        "--hidden_dims",
        default=[128],
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
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the CIFAR10 dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_info = train(**kwargs)

    # feel free to comment in to see plots
    # # print("Test accuracy: ", test_accuracy)
    # # print("f1 score: ", logging_info["test_metrics"]["f1_beta"])

    # train_loss = logging_info["train_loss"]
    # val_losses = logging_info["val_loss"]
    # test_metrics = logging_info["test_metrics"]

    # plt.plot(train_loss, label="train_loss")
    # plt.plot(val_losses, label="val_loss")
    # plt.legend()
    # plt.show()

    # plt.plot(val_accuracies, label="val_acc")
    # plt.plot(logging_info["train_acc"], label="train_acc")
    # plt.legend()

    # plt.matshow(test_metrics["conf_mat"])

    # # add legend to confusion matrix
    # plt.colorbar()
    # # add axis labels to confusion matrix
    # plt.xlabel("predicted")
    # plt.ylabel("true")
    # plt.show()

    # print(val_accuracies)
    # print(test_accuracy)

    # # convert confusion matrix to metrics with beta 0.1, 1 and 10 and print f1 scores
    # print(confusion_matrix_to_metrics(test_metrics["conf_mat"], beta=0.1)["f1_beta"])
    # print(confusion_matrix_to_metrics(test_metrics["conf_mat"], beta=1)["f1_beta"])
    # print(confusion_matrix_to_metrics(test_metrics["conf_mat"], beta=10)["f1_beta"])
