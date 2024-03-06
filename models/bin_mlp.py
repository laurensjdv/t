import torch.nn as nn
import torch.nn.functional as F
import torch


class binMLP(nn.Module):
    def __init__(self, n_inputs, n_hidden, dropout=0.5):
        super().__init__()
        self.nn = nn.ModuleList()

        for i in range(len(n_hidden)):
            if i == 0:
                self.nn.append(nn.Linear(n_inputs, n_hidden[i]))
            else:
                self.nn.append(nn.Linear(n_hidden[i - 1], n_hidden[i]))
            self.nn.append(nn.ReLU())
            self.nn.append(nn.Dropout(dropout))

        # Append inputs if no hidden layers
        if len(n_hidden) == 0:
            self.nn.append(nn.Linear(n_inputs, 1))
        else:
            self.nn.append(nn.Linear(n_hidden[-1], 1))

    def forward(self, x):
        out = x
        for module in self.nn[:-1]:  # Apply all modules except the last Linear layer
            out = module(out)
        out = torch.sigmoid(
            self.nn[-1](out)
        )  # Apply sigmoid to the output of the last Linear layer
        return out

    def l1_loss(self, l1_lambda):
        l1 = 0
        for p in self.parameters():
            l1 = l1 + p.abs().sum()
        return l1 * l1_lambda

    def l2_loss(self, l2_lambda):
        l2 = 0
        for p in self.parameters():
            l2 = l2 + p.pow(2).sum()
        return l2 * l2_lambda
