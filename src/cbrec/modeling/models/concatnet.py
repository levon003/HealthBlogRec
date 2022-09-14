"""
ConcatNet is not yet implemented.
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import cbrec.modeling.modelconfig


class ConcatNet(nn.Module):
    """
    Network that manages source and candidate features separately.
    
    In general, this model resembles a "bi-encoder" (aka Siamese/"twin tower" architecture). 
    Sentence-BERT uses different training and inference architectures.
    For training, it takes torch.cat((a1, a2, |a1 - a2|))
    Inteference uses cosine similarity: cos(a1, a2)
    
    TODO add ConcatNet_include_shared config value to concatenate the shared features into the representation. Also, especially if handling the shared features separately, they really should be converted to a single categorical feature that we then embed.

    """
    def __init__(self, 
                     model_config: cbrec.modeling.modelconfig.ModelConfig,
                     feature_manager: cbrec.modeling.preprocess.FeatureManager):
        super(ConcatNet, self).__init__()
                
        n_input = model_config.ConcatNet_n_input
        n_hidden = model_config.ConcatNet_n_hidden
        dropout_p = model_config.ConcatNet_dropout_p
                
        # note: 768 is the size of the roBERTa outputs
        self.source_inds = np.array(feature_manager.get_feature_indices('source', '*'))
        self.candidate_inds = np.array(feature_manager.get_feature_indices('candidate', '*'))
        self.fc1_source = nn.Linear(self.source_inds.size, n_hidden)
        self.fc1_candidate = nn.Linear(self.candidate_inds.size, n_hidden)
        self.fc2 = nn.Linear(2*n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1, bias=False)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        
        
    def forward(self, x):
        source = x[:,self.source_inds]
        candidate = x[:,self.candidate_inds]
        source = F.relu(self.fc1_source(source))
        candidate = F.relu(self.fc1_candidate(candidate))
        source = self.dropout1(source)
        candidate = self.dropout1(candidate)
        combine = torch.cat((source,candidate),1)  # TODO add flag to also include torch.abs(source - candidate) to the combined matrix
        x = F.relu(self.fc2(combine))
        x = self.dropout2(x)
        x = self.fc3(x)  # note: not using F.sigmoid here, as the loss used includes the Sigmoid transformation
        return x
    
    
def emb_sz_rule(n_cat):
    """Rule of thumb to pick embedding size corresponding to `n_cat`.
    
       Source: https://github.com/fastai/fastai/blob/master/nbs/42_tabular.model.ipynb
    
       Copyright 2021 fast.ai, Licensed under the Apache License, Version 2.0
    """
    return min(600, round(1.6 * n_cat**0.56))
