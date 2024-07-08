import numpy as np
import pandas as pd
import argparse
import os.path as osp
import os
import pickle

import torch
import torch.nn.functional as func
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import svm


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

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

def get_metrics(pred, label):
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    auc_score = roc_auc_score(label, pred)
    epoch_acc = (tn + tp) / (tn + tp + fn + fp)
    return auc_score, epoch_acc


def run_experiment(ds, seed, data_dir, udi, config=None):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    run = wandb.init(config=config)

    C = run.config["C"]
    gamma = run.config["gamma"]
    kernel = run.config["kernel"]
    degree = run.config["degree"]
    mrmr = run.config["mrmr"]

    print(run.name)
    print(f"mrmr: {mrmr}")

    if ds == "data/csv/severe_rds_v2.csv":
        mrmr_features = np.array([1140, 689, 427, 122, 139, 765, 907, 1384, 22, 1293, 1449, 492, 1440, 499, 1316, 1318, 135, 879, 886, 223, 1455, 676, 1136, 1464, 70, 1462, 1386, 939, 111, 413, 10, 1403, 1027, 547, 395, 1210, 942, 501, 45, 425, 638, 505, 26, 1409, 1448, 605, 232, 1459, 821, 1394, 1376, 809, 1163, 216, 791]
            )
    elif ds == "data/csv/rds_under_8_severe_v2.csv":
        mrmr_features = np.array([1035, 382, 967, 1316, 34, 852, 761, 480, 875, 1357, 1203, 686, 415, 248, 1230, 1370, 1275, 466, 1274, 126, 840, 1470, 448, 629, 1292, 922, 617, 168, 946, 1372, 131, 219, 1247, 413, 72, 496, 880, 275, 863, 11, 571, 147, 877, 843, 92, 124, 1157, 1457, 249, 1126, 741, 1289, 1270, 1453, 742]
            )
    else:
        mrmr_features = np.array([1140, 536, 223, 907, 1449, 499, 1293, 45, 135, 1440, 879, 1384, 1210, 1316, 122, 22, 492, 638, 765, 1027, 1464, 501, 1462, 395, 26, 1079, 70, 425, 1403, 1409, 1318, 886, 1459, 1448, 939, 1163, 547, 10, 413, 676, 131, 216, 942, 1136, 1386, 232, 1455, 1337, 814, 139, 392, 1376, 1382, 471, 656]
        )


    if mrmr is True:
        input_features = 55
        # mrmr_features = np.array([1140, 536, 223, 907, 1449, 499, 1293, 45, 135, 1440, 879, 1384, 1210, 1316, 122, 22, 492, 638, 765, 1027, 1464, 501, 1462, 395, 26, 1079, 70, 425, 1403, 1409, 1318, 886, 1459, 1448, 939, 1163, 547, 10, 413, 676, 131, 216, 942, 1136, 1386, 232, 1455, 1337, 814, 139, 392, 1376, 1382, 471, 656]
        # )
    else:
        input_features = 1485
        mrmr_features = None


    print(f"Using {input_features} features")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=99)

    labels = np.genfromtxt(ds)
    print(DEVICE)
    eids = labels[:,0]

    labels = labels[:,1]

    eval_metrics = np.zeros((skf.n_splits, 2))

    data_dir = data_dir + "/raw"
    dataset = FCMatrixDataset(ds, data_dir, udi, None, mrmr=mrmr_features)

    for n_fold, (train_val, test) in enumerate(skf.split(labels, labels)):
        
        if kernel == "poly":
            clf = svm.SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)
        else:
            clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)

        train_val_dataset, test_dataset = Subset(dataset, train_val), Subset(dataset, test)
        train_val_labels = labels[train_val]
        train_val_index = np.arange(len(train_val_dataset))

        train, val, _, _ = train_test_split(train_val_index, train_val_labels, test_size=0.11, shuffle=True, stratify=train_val_labels)
        train_dataset, val_dataset = Subset(train_val_dataset, train), Subset(train_val_dataset, val)
        
        def seperate_subset(subset):
            input = [np.array(subset[i][0]) for i in range(len(subset))]
            input = np.array(input)
            label = [subset[i][1] for i in range(len(subset))]
            label = np.array(label)
            return input, label

        train_input, train_label = seperate_subset(train_dataset)
        val_input, val_label = seperate_subset(val_dataset)
        test_input, test_label = seperate_subset(test_dataset)

        clf.fit(train_input, train_label)

        val_pred = clf.predict(val_input)
        val_auc, val_acc = get_metrics(val_pred, val_label)

        test_pred = clf.predict(test_input)
        test_auc, test_acc = get_metrics(test_pred, test_label)

        print('CV: {:03d},  Test BAC: {:.5f}, TEST AUC: {:.5f} VAL BAC: {:.5f}, VAL AUC: {:.5f}'.format(n_fold + 1,
            test_auc,test_acc, val_auc, val_acc))
        
        run.log({"n_fold": n_fold,"val_auc": val_auc, "val_acc":val_acc, "test_auc": test_auc, "test_acc": test_acc})

        eval_metrics[n_fold, 0] = test_auc
        eval_metrics[n_fold, 1] = test_acc

    eval_df = pd.DataFrame(eval_metrics)
    eval_df.columns = ['AUC', 'ACC']
    eval_df.index = ['Fold_%02i' % (i + 1) for i in range(skf.n_splits)]
    print(eval_df)
    print('Average AUC: %.4f±%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
    print('Average Accuracy: %.4f±%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
    run.log({"AUC": eval_metrics[:, 0].mean(), "ACC": eval_metrics[:, 1].mean(), "ACC std": eval_metrics[:, 1].std(), "AUC std": eval_metrics[:, 0].std()})

    run_name = run.name

    # save the model to models/saved/SVM
    model_path = f"models/saved/SVM/{run_name}.pkl"
    with open(model_path, 'wb') as f:
        torch.save(clf, f)
    


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--dataset",
        default="data/csv/severe_rds.csv",
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

    os.environ['WANDB_AGENT_DISABLE_FLAPPING'] = 'true'
    os.environ['FLAPPING_MAX_FAILURES'] = '100'
    os.environ['WANDB_AGENT_MAX_INITIAL_FAILURES'] = '100'

    
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
    sweep_id = wandb.sweep(sweep_config, project=f"{dataset_name}_SVM")

    wandb.agent(sweep_id, train_partial, count=400)