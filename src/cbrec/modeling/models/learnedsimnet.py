

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import cbrec.modeling.modelconfig
import cbrec.modeling.preprocess


class LearnedSimNet(nn.Module):
    """
    Similarity-based net with a learned embedding representation of the source and target.
    
    The non-shared, cosine-similarity version resembles DSSM (Deep Structured Semantic Model), introduced by Huang et al. (2013).  Discussed in "4.1.2 Representing with Feedforward Neural Networks" in the Xu et al. (2020) book.
    
    This is highly similar to the Sentence-BERT bi-encoder design, but instead of BERT we learn a linear layer or two as the "encodings".
    In general, though, we can think of this model as likely worse than cross-encoder designs based on Lin et al 2020's summary.
    
    If the inputs were a "bag of interactions", I believe this model is very close to Deep Matrix Factorization (DeepMF, Xue et al. 2017).
    From that perspective, the inputs are a row (initiator) and column (receiver) of the author-author interaction matrix, with a 1 if an initiation occurred and a 0 otherwise.
    Quoting Xu et al.'s book (2020): "As the spaces of users and items are different, DeepMF uses two MLPs with different parameters to represent users and items."
    In our context, that suggests that we might use a single MLP to represent authors OR two MLPs to distiniguish between initiators and receivers.
    Restating, LearnedSimNet is DeepMF but using user content vectors rather than "bag of interactions".
    (My thought: can easily identify the sparse initiations needed to build the "bag of interactions" from the metadata dictionaries.)
    
    """
    def __init__(self, 
                    model_config: cbrec.modeling.modelconfig.ModelConfig,
                    feature_manager: cbrec.modeling.preprocess.FeatureManager
                ):
        super(LearnedSimNet, self).__init__()
        
        if model_config.LearnedSimNet_similarity_function == 'cosine':
            self.similarity_function = torch.nn.CosineSimilarity(dim=1)
        elif model_config.LearnedSimNet_similarity_function == 'l2':
            self.similarity_function = torch.nn.PairwiseDistance(p=2)
        else:
            raise ValueError(f"Unknown similarity function '{model_config.LearnedSimNet_similarity_function}'.")
            
        self.source_inds = np.array(feature_manager.get_feature_indices('source', '*'))
        self.candidate_inds = np.array(feature_manager.get_feature_indices('candidate', '*'))
        assert self.source_inds.shape == self.candidate_inds.shape, "Expected source and candidate to have same feature count."
        n_input = len(self.source_inds)
        
        self.share_layers = model_config.LearnedSimNet_share_layers
        n_hidden_source = model_config.LearnedSimNet_n_hidden_source
        n_hidden_candidate = model_config.LearnedSimNet_n_hidden_candidate
        assert n_hidden_source == n_hidden_candidate, "Different sizes for source and candidate embedding sizes is not yet supported."
        
        self.fc_source = nn.Linear(n_input, n_hidden_source)
        if not self.share_layers:
            self.fc_candidate = nn.Linear(n_input, n_hidden_candidate)
            
        self.include_shared = model_config.LearnedSimNet_include_shared
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
        source_x = F.sigmoid(self.fc_source(source_x))
        if self.share_layers:
            candidate_x = F.sigmoid(self.fc_source(candidate_x))
        else:
            candidate_x = F.sigmoid(self.fc_candidate(candidate_x))
        if self.include_shared:
            # totally untested code path
            shared_x = x[:,self.shared_inds]
            sim = self.similarity_function(source_x, candidate_x).reshape((-1, 1))
            x = torch.cat((shared_x, sim), 1)
            return self.fc_shared(x)
        else:
            return self.similarity_function(source_x, candidate_x).reshape((-1, 1))
