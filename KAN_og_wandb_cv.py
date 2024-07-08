from kan import *

import numpy as np
import pandas as pd
import argparse
import os.path as osp

import torch
import torch.nn.functional as func
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

import wandb
import yaml

from models.mlp import MLP
from models.bin_mlp import binMLP
# from dataloaders.batch_dataloader import FCMatrixDataset
from dataloaders.dataloader import FCMatrixDataset

from torch.utils.data import Dataset, DataLoader, Subset

from utils import balanced_random_split_v2
from copy import deepcopy
from functools import partial

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def KAN_train(model, loader, train_dataset, batch_size, optimizer):
    loss_all = 0
    for data in loader:
        inputs = data[0].to(DEVICE)
        labels = data[1].to(DEVICE)

        output = model(inputs)
       
        loss = func.cross_entropy(output, labels)
        loss_all += batch_size * loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_all / len(train_dataset)


def KAN_test(model, loader, datas, batch_size, show_pred=False):
    pred = []
    label = []
    loss_all = 0
    for data in loader:
        inputs = data[0].to(DEVICE)
        labels = data[1].to(DEVICE)

        output = model(inputs)
        loss = func.cross_entropy(output, labels)
        loss_all += batch_size * loss.item()
        _, predicted = torch.max(output, 1)
        pred.append(predicted)
        label.append(labels)

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()
    tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
    auc_score = roc_auc_score(y_true, y_pred)

    epoch_acc = (tn + tp) / (tn + tp + fn + fp)
    if show_pred is True:
        print(y_pred)
    return auc_score, epoch_acc, loss_all / len(datas)


def run_experiment(ds, seed, data_dir, udi, config=None):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False


    run = wandb.init(config=config)

    batch_size = run.config["batch_size"]
    n_epochs = run.config["epochs"]
    layer_size = run.config["layer_size"]
    hidden_dim_ratio = run.config["hidden_dim_ratio"]
    n_layers = run.config["n_layers"]
    lr = run.config["learning_rate"]
    mrmr = run.config["mrmr"]
    print(run.name)
    print(f"mrmr: {mrmr}")


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


    print(f"Using {input_features} features")
    hidden_dims = [int(layer_size * hidden_dim_ratio ** i) for i in range(n_layers)]
    dims = [input_features] + hidden_dims + [2]
    print(dims)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)

    labels = np.genfromtxt(ds)
    print(DEVICE)
    eids = labels[:,0]

    labels = labels[:,1]

    eval_metrics = np.zeros((skf.n_splits, 2))

    data_dir = data_dir + "/raw"
    dataset = FCMatrixDataset(ds, data_dir, udi, None, mrmr=mrmr_features)

    for n_fold, (train_val, test) in enumerate(skf.split(labels, labels)):
        
        model = KAN(dims, device=DEVICE, symbolic_enabled=False, bias_trainable=False)
        print(model)

        for layer in model.act_fun:
            print(layer.in_dim, layer.out_dim)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_val_dataset, test_dataset = Subset(dataset, train_val), Subset(dataset, test)
        train_val_labels = labels[train_val]
        train_val_index = np.arange(len(train_val_dataset))

        train, val, _, _ = train_test_split(train_val_index, train_val_labels, test_size=0.11, shuffle=True, stratify=train_val_labels)
        train_dataset, val_dataset = Subset(train_val_dataset, train), Subset(train_val_dataset, val)
       
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        min_v_loss = np.inf
        best_val_acc = 0
        for epoch in range(n_epochs):
            t_loss = KAN_train(model, train_loader, train_dataset, batch_size, optimizer)
            _, val_acc, v_loss = KAN_test(model, val_loader, val_dataset, batch_size)

            if min_v_loss > v_loss:
                min_v_loss = v_loss
                best_val_acc = val_acc
                best_model = KAN(dims, device=DEVICE)
                best_model.load_state_dict(model.state_dict())
            if n_fold == 0:
                run.log({"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss, "val_acc": val_acc})
                
            print('CV: {:03d}, Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}, Val ACC: {:.5f}'.format(n_fold + 1, epoch + 1, t_loss, v_loss, val_acc))

        test_auc, test_acc, _ = KAN_test(best_model, test_loader, test_dataset, batch_size, show_pred=True)
        print('CV: {:03d}, Epoch: {:03d}, Val Loss: {:.5f}, Val BAC: {:.5f}, Test BAC: {:.5f}, TEST AUC: {:.5f}'.format(n_fold + 1, epoch + 1, min_v_loss, best_val_acc, test_auc,
                                                test_acc))
        
        run.log({"n_fold": n_fold, "test_auc": test_auc, "test_acc": test_acc})


        eval_metrics[n_fold, 0] = test_auc
        eval_metrics[n_fold, 1] = test_acc

    eval_df = pd.DataFrame(eval_metrics)
    eval_df.columns = ['AUC', 'ACC']
    eval_df.index = ['Fold_%02i' % (i + 1) for i in range(skf.n_splits)]
    print(eval_df)
    print('Average AUC: %.4f±%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
    print('Average Accuracy: %.4f±%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
    run.log({"AUC": eval_metrics[:, 0].mean(), "ACC": eval_metrics[:, 1].mean(), "ACC std": eval_metrics[:, 1].std(), "AUC std": eval_metrics[:, 0].std()})

    # save model
    # get the name of the run
    run_name = run.name

    torch.save(best_model.state_dict(), f"models/saved/slowKAN/{run_name}.pth")

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
        default="data/fetched/25751",
        type=str,
        help="Data directory where to find the dataset.",
    )
    parser.add_argument(
        "--sweep_config",
        default="adam_config.yaml",
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
    dataset = kwargs["dataset"][0]

    with open(sweep_config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    train_partial = partial(run_experiment, seed=kwargs["seed"], ds=dataset, data_dir=kwargs["data_dir"], udi=udi)

    print(sweep_config)

    dataset_name = dataset.split("/")[-1].split(".")[0]

    sweep_id = wandb.sweep(sweep_config, project=f"{dataset_name}_slowKAN_{sweep_config['name']}")
    wandb.agent(sweep_id, train_partial, count=100)