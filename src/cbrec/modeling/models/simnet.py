

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import cbrec.modeling.modelconfig
import cbrec.modeling.preprocess


class SimNet(nn.Module):
    """
    No-parameter similarity-based net.
    
    
    """
    def __init__(self, 
                    model_config: cbrec.modeling.modelconfig.ModelConfig,
                    feature_manager: cbrec.modeling.preprocess.FeatureManager
                ):
        super(SimNet, self).__init__()
        
        if model_config.SimNet_similarity_function == 'cosine':
            self.similarity_function = torch.nn.CosineSimilarity(dim=1)
        elif model_config.SimNet_similarity_function == 'l2':
            self.similarity_function = torch.nn.PairwiseDistance(p=2)
        else:
            raise ValueError(f"Unknown similarity function '{model_config.SimNet_similarity_function}'.")
            
        self.source_inds = np.array(feature_manager.get_feature_indices('source', '*'))
        self.candidate_inds = np.array(feature_manager.get_feature_indices('candidate', '*'))
        assert self.source_inds.shape == self.candidate_inds.shape, "Expected source and candidate to have same feature count."
        print(self.source_inds.shape)
        
    def forward(self, x):
        assert x.shape[1] == len(self.source_inds) + len(self.candidate_inds)
        source_x = x[:,self.source_inds]
        candidate_x = x[:,self.candidate_inds]
        #assert source_x.shape == candidate_x.shape, f"{x.shape} = {n_source} {source_x.shape} {candidate_x.shape}"
        return self.similarity_function(source_x, candidate_x).reshape((-1, 1))