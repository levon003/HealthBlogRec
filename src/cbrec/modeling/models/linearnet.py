

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import cbrec.modeling.modelconfig


class LinearNet(nn.Module):
    """
    Simple neural net with 2 hidden layers.
    """
    def __init__(self, 
                     model_config: cbrec.modeling.modelconfig.ModelConfig):
        super(LinearNet, self).__init__()
        
        n_input = model_config.LinearNet_n_input
        n_hidden = model_config.LinearNet_n_hidden
        dropout_p = model_config.LinearNet_dropout_p
        
        # note: 768 is the size of the roBERTa outputs
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1, bias=False)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # note: not using F.sigmoid here, as the loss used includes the Sigmoid transformation
        return x