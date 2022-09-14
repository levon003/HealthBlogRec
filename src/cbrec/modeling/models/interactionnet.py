

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import cbrec.modeling.modelconfig
import cbrec.modeling.preprocess


class InteractionNet(nn.Module):
    """
    Explicitly model interactions between the source and target.
    
    """
    def __init__(self, 
                    model_config: cbrec.modeling.modelconfig.ModelConfig,
                    feature_manager: cbrec.modeling.preprocess.FeatureManager
                ):
        super(InteractionNet, self).__init__()
        
            
        self.source_inds = np.array(feature_manager.get_feature_indices('source', '*'))
        self.candidate_inds = np.array(feature_manager.get_feature_indices('candidate', '*'))
        assert self.source_inds.shape == self.candidate_inds.shape, "Expected source and candidate to have same feature count."
        
        n_input = len(self.source_inds)
        self.preint_n_hidden = model_config.InteractionNet_preint_n_hidden
        if self.preint_n_hidden > 0:
            self.fc_source = nn.Linear(n_input, self.preint_n_hidden)
            self.fc_candidate = nn.Linear(n_input, self.preint_n_hidden)
            n_input = self.preint_n_hidden
        
        self.fc_ints = nn.Linear(n_input * n_input, 100)
        self.fc_final = nn.Linear(100, 1, bias=False)
        
        self.include_shared = model_config.InteractionNet_include_shared
        if self.include_shared:
            self.shared_inds = np.array(feature_manager.get_feature_indices('shared', '*'))
            n_shared = self.shared_inds
            # this layer takes the shared features and the similarity score
            self.fc_shared = nn.Linear(1 + n_shared, 1, bias=False)
        
    def forward(self, x):
        if self.include_shared:
            assert x.shape[1] == len(self.source_inds) + len(self.candidate_inds) + len(self.shared_inds)
        else:
            assert x.shape[1] == len(self.source_inds) + len(self.candidate_inds)
        source_x = x[:,self.source_inds]
        candidate_x = x[:,self.candidate_inds]
        
        if self.preint_n_hidden > 0:
            # apply a linear layer here to reduce the size of the inputs before computing interactions
            source_x = F.relu(self.fc_source(source_x))
            candidate_x = F.relu(self.fc_candidate(candidate_x))
        
        source_x = source_x.reshape((source_x.shape[0], source_x.shape[1], 1))
        candidate_x = candidate_x.reshape((candidate_x.shape[0], 1, candidate_x.shape[1]))
        ints = torch.matmul(source_x, candidate_x)
        assert ints.shape == (x.shape[0], len(self.source_inds), len(self.candidate_inds))

        # TODO consider an attention layer or something similar here as well
        
        ints = ints.reshape(x.shape[0], self.fc_ints.in_features)
        ints = F.relu(self.fc_ints(ints))
        return self.fc_final(ints)
        
        
        #if self.include_shared:
        #    TODO do this before running through fc_ints
        #    shared_x = x[:,self.shared_inds]
        #    sim = self.similarity_function(source_x, candidate_x).reshape((-1, 1))
        #    x = torch.cat((shared_x, sim), 1)
        #    return self.fc_shared(x)
