import torch.nn as nn
import torch.nn.functional as F
import torch

class binMLP(nn.Module):
    def __init__(self, n_inputs, n_hidden, dropout=0.5, weight_constraint=1.0):
        super().__init__()
        
        self.nn = nn.ModuleList()
        
        for i in range(len(n_hidden)):
            if i == 0:
                self.nn.append(nn.Linear(n_inputs, n_hidden[i]))
            else:
                self.nn.append(nn.Linear(n_hidden[i - 1], n_hidden[i]))
            self.nn.append(nn.ReLU())
        
        # Adjusting the output layer for binary classification
        self.nn.append(nn.Linear(n_hidden[-1], 1))

    def forward(self, x):
        out = x
        for module in self.nn[:-1]:  # Apply all modules except the last Linear layer
            out = module(out)
        out = torch.sigmoid(self.nn[-1](out))  # Apply sigmoid to the output of the last Linear layer
        return out