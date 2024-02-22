from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_classes):
        super().__init__()

        self.nn = nn.ModuleList()
        
        for i in range(len(n_hidden)):
            if i == 0:
                self.nn.append(nn.Linear(n_inputs, n_hidden[i]))
            else:
                self.nn.append(nn.Linear(n_hidden[i - 1], n_hidden[i]))
            self.nn.append(nn.ReLU())

        self.nn.append(nn.Linear(n_hidden[-1], n_classes))

    def forward(self, x):
        out = x
        for module in self.nn:
            out = module(out)
        return out

    