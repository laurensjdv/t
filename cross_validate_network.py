import numpy as np
import pandas as pd
import os.path as osp

import torch
import torch.nn.functional as func
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix

from models.mlp import MLP
from models.bin_mlp import binMLP
# from dataloaders.batch_dataloader import FCMatrixDataset
from dataloaders.dataloader import FCMatrixDataset

from torch.utils.data import Dataset, DataLoader, Subset


from utils import balanced_random_split_v2


def MLP_train(loader):
    model.train()

    loss_all = 0
    for data in loader:
        inputs = data[0].to(device)
        labels = data[1].to(device)

        optimizer.zero_grad()
        output = model(inputs)
       
        loss = func.cross_entropy(output, labels)
        loss.backward()
        loss_all += batch_size * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def MLP_test(loader):
    model.eval()

    pred = []
    label = []
    loss_all = 0
    for data in loader:
        inputs = data[0].to(device)
        labels = data[1].to(device)

        output = model(inputs)
        loss = func.cross_entropy(output, labels)
        loss_all += batch_size * loss.item()
        # pred.append(func.softmax(output, dim=1).max(dim=1)[1])
        _, predicted = torch.max(output, 1)
        pred.append(predicted)
        label.append(labels)

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()
    tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
    epoch_sen = tp / (tp + fn)
    epoch_spe = tn / (tn + fp)
    epoch_acc = (tn + tp) / (tn + tp + fn + fp)
    return epoch_sen, epoch_spe, epoch_acc, loss_all / len(val_dataset)


batch_size = 240
dropout = 0.3637538
n_epochs = 25
hidden_dims = [224]
lr = 0.0009
weight_decay = 0.001

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=99)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset = 'data/csv/matched_smith_single_ep.csv'
# dataset = 'data/csv/matched_smith_single_ep_rds_over_8.csv'
# dataset = 'data/csv/matched_smith_severe.csv'
dataset = 'data/csv/matched_smith_severe_rds_over_8.csv'
# dataset = 'data/csv/matched_smith_moderate.csv'
# dataset = 'data/csv/matched_smith_moderate_rds_over_8.csv'
# dataset= 'data/csv/matched_rds_over_8.csv'
# dataset = 'data/csv/matched_rds_HC_smith_single_ep.csv'
# dataset = 'data/csv/matched_rds_HC_smith_severe.csv'
# dataset = 'data/csv/matched_rds_HC_smith_moderate.csv'
# dataset = 'data/csv/balanced_sex_classification.csv'
labels = np.genfromtxt(dataset)
print(labels)
print(device)
eids = labels[:,0]

labels = labels[:,1]
# count values in eids
# print(len(set(eids)))
print(eids)
eval_metrics = np.zeros((skf.n_splits, 3)
                        )
input_features = 55
mrmr_features = np.array([1140, 536, 223, 907, 1449, 499, 1293, 45, 135, 1440, 879, 1384, 1210, 1316, 122, 22, 492, 638, 765, 1027, 1464, 501, 1462, 395, 26, 1079, 70, 425, 1403, 1409, 1318, 886, 1459, 1448, 939, 1163, 547, 10, 413, 676, 131, 216, 942, 1136, 1386, 232, 1455, 1337, 814, 139, 392, 1376, 1382, 471, 656]
    )

dataset = FCMatrixDataset(dataset, 'data/fetched/25751/raw', '25751', mapping=None, mrmr=mrmr_features)

print(len(dataset))
for n_fold, (train_val, test) in enumerate(skf.split(labels, labels)):
    
    model = MLP(55, hidden_dims, 2, dropout).to(device)
    # model = GCN(55, hidden_dims, 2, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ## OG CODE ##
    # train_val_dataset, test_dataset = dataset[train_val.tolist()], dataset[test.tolist()]
    # train_val_labels = labels[train_val]
    # train_val_index = np.arange(len(train_val_dataset))

    # train, val, _, _ = train_test_split(train_val_index, train_val_labels, test_size=0.11, shuffle=True, stratify=train_val_labels)
    # train_dataset, val_dataset = train_val_dataset[train.tolist()], train_val_dataset[val.tolist()]
    ## //OG CODE ##
    
    train_val_dataset, test_dataset = Subset(dataset, train_val), Subset(dataset, test)
    train_val_labels = labels[train_val]
    train_val_index = np.arange(len(train_val_dataset))

    train, val, _, _ = train_test_split(train_val_index, train_val_labels, test_size=0.11, shuffle=True, stratify=train_val_labels)
    train_dataset, val_dataset = Subset(train_val_dataset, train), Subset(train_val_dataset, val)
    # total_size = len(dataset)
    # train_size = int(0.8 * total_size)
    # val_size = int(0.1 * total_size)

    # test_size = total_size - train_size - val_size
    
    # train_dataset, val_dataset, test_dataset = balanced_random_split_v2(dataset, [train_size, val_size, test_size], num_classes=2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    min_v_loss = np.inf
    best_val_acc = 0
    no_improvement_epochs = 0
    for epoch in range(n_epochs):
        t_loss = MLP_train(train_loader)
        val_sen, val_spe, val_acc, v_loss = MLP_test(val_loader)
        test_sen, test_spe, test_acc, _ = MLP_test(test_loader)

        no_improvement_epochs += 1

        if min_v_loss > v_loss:
        # if best_val_acc < val_acc:
            min_v_loss = v_loss
            best_val_acc = val_acc
            best_test_sen, best_test_spe, best_test_acc = test_sen, test_spe, test_acc
            # torch.save(model.state_dict(), 'best_model_%02i.pth' % (n_fold + 1))
            no_improvement_epochs = 0

        if no_improvement_epochs > 10:
            break
    
    print('CV: {:03d}, Epoch: {:03d}, Val Loss: {:.5f}, Val BAC: {:.5f}, Test BAC: {:.5f}, TEST SEN: {:.5f}, '
                  'TEST SPE: {:.5f}'.format(n_fold + 1, epoch + 1, min_v_loss, best_val_acc, best_test_acc,
                                            best_test_sen, best_test_spe))

    eval_metrics[n_fold, 0] = best_test_sen
    eval_metrics[n_fold, 1] = best_test_spe
    eval_metrics[n_fold, 2] = best_test_acc

eval_df = pd.DataFrame(eval_metrics)
eval_df.columns = ['SEN', 'SPE', 'ACC']
eval_df.index = ['Fold_%02i' % (i + 1) for i in range(skf.n_splits)]
print(eval_df)
print('Average Sensitivity: %.4f±%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
print('Average Specificity: %.4f±%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
print('Average Accuracy: %.4f±%.4f' % (eval_metrics[:, 2].mean(), eval_metrics[:, 2].std()))
