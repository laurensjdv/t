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

from models.GCN import GCN
from dataloaders.GCN_loader import GraphDataset

# from dataloaders.batch_dataloader import FCMatrixDataset
from dataloaders.dataloader import FCMatrixDataset

from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader


from utils import balanced_random_split_v2
from copy import deepcopy
from functools import partial



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
def GCN_train(model, loader, train_dataset, batch_size, optimizer):
    model.train()

    loss_all = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = func.cross_entropy(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def GCN_test(model, loader, val_dataset, batch_size):
    model.eval()

    pred = []
    label = []
    loss_all = 0
    for data in loader:
        data = data.to(DEVICE)
        output = model(data)
        loss = func.cross_entropy(output, data.y)
        loss_all += data.num_graphs * loss.item()
        pred.append(func.softmax(output, dim=1).max(dim=1)[1])
        label.append(data.y)

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()
    tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
    auc_score = roc_auc_score(y_true, y_pred)

    epoch_acc = (tn + tp) / (tn + tp + fn + fp)
    return auc_score, epoch_acc, loss_all / len(val_dataset)


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
    dropout = run.config["dropout"]
    n_epochs = run.config["epochs"]
    # layer_size = run.config["layer_size"]
    # n_layers = run.config["n_layers"]
    k_degree = run.config["k_degree"]
    k_order = run.config["k_order"]
    lr = run.config["learning_rate"]
    mrmr = run.config["mrmr"]

    print(mrmr)

    # if mrmr is True:
    #     input_features = 11
    #     mrmr_features = np.array([1140, 536, 223, 907, 1449, 499, 1293, 45, 135, 1440, 879, 1384, 1210, 1316, 122, 22, 492, 638, 765, 1027, 1464, 501, 1462, 395, 26, 1079, 70, 425, 1403, 1409, 1318, 886, 1459, 1448, 939, 1163, 547, 10, 413, 676, 131, 216, 942, 1136, 1386, 232, 1455, 1337, 814, 139, 392, 1376, 1382, 471, 656]
    #     )
    # else:
    #     input_features = 55
    #     mrmr_features = None

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
        input_features = 11
        print("regular mrmr features")
    elif mrmr == "sort":
        input_features = 11
        mrmr_features = np.sort(mrmr_features)
        print("sorted mrmr features")
    else:
        input_features = 55
        mrmr_features = None

    print(f"Using {input_features} features")


    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=99)


    labels = np.genfromtxt(ds)
    print(DEVICE)
    eids = labels[:,0]

    labels = labels[:,1]

    eval_metrics = np.zeros((skf.n_splits, 2))


    # data_dir = data_dir + "/raw"
    dataset = GraphDataset(ds, data_dir, udi, None, mrmr = mrmr_features)

    dataset.process(k_degree)


    for n_fold, (train_val, test) in enumerate(skf.split(labels, labels)):
        
        model = GCN(input_features, 2, k_order, dropout).to(DEVICE)
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
            t_loss = GCN_train(model, train_loader, train_dataset, batch_size, optimizer)
            _, val_acc, v_loss = GCN_test(model, val_loader, val_dataset, batch_size)

            if min_v_loss > v_loss:
            # if best_val_acc < val_acc:
                min_v_loss = v_loss
                best_val_acc = val_acc
                best_model = deepcopy(model)
            if n_fold == 0:
                run.log({"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss, "val_acc": val_acc})

        test_auc, test_acc, _ = GCN_test(best_model, test_loader, test_dataset, batch_size)
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

    run_name = run.name
    torch.save(model.state_dict(), f"models/saved/GCN/{run_name}.pth")


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

    sweep_id = wandb.sweep(sweep_config, project=f"{dataset_name}_GCN")
    wandb.agent(sweep_id, train_partial, count=200)