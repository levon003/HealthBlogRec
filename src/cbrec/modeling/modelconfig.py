
import json
from copy import deepcopy

"""
The default_values_dict defines the keys and default values expected to exist in a ModelConfig instance.

It is safe to write code that assumes the existence of the keys provided in the default_values_dict.
It is NOT safe to write code that assumes the existence of keys NOT in the default_values_dict.
Note, however, that loading an old config might result in confusing combinations of old defaults and new keys.
"""
default_values_dict = {
    'model_name': 'LinearNet',
    'experiment_name': 'default',  # if generating configs in a batch e.g. for optimizing hyperparameters, set experiment_name to something semantically useful. Note this gets put into the output filepath.
    
    'output_dir': '/home/lana/shared/caringbridge/data/projects/recsys-peer-match/torch_experiments/modeling',
    'output_name': 'default',  # uses datetime
    'output_basename': None,  # set by ModelManager.__init__
    
    'preprocess_use_scaler': True,
    'preprocess_drop_columns': [], # must be column keys present in data config
    'preprocess_encode_columns': [],  # example [{"original_column_name": "bla", "bins": [0, 7]}],
    
    'LinearNet_n_input': -1,  # set this pre-instantiation
    'LinearNet_n_hidden': 100,
    'LinearNet_dropout_p': 0.1,
    
    'ConcatNet_n_input':-1,
    'ConcatNet_n_hidden': 100,
    'ConcatNet_dropout_p': 0.1,
    'ConcatNet_text_feature_columns': [],
    
    'SimNet_similarity_function': 'cosine',
    
    'train_verbose': True,
    'train_n_epochs': 1000,
    'train_lr_init': 0.01,
    'train_validation_rate': 0.1,  # (vr) we will compute loss and accuracy against the validation set on vr of the epochs
    'train_weight_decay': 0.01,
    
    'train_Adam_beta1': 0.9,
    'train_Adam_beta2': 0.999,
    'train_Adam_eps': 1e-08,
    
    'train_scheduler_name': 'OneCycleLR',
    'train_max_lr': 0.0155,
    
    
    'data_version': "1.0.0",
}

class ModelConfig:
    def from_dict(config_dict):
        config = ModelConfig()
        config.update_from_dict(config_dict)
        return config
    
    
    def from_filepath(filepath):
        with open(filepath, 'r') as infile:
            config_dict = json.loads(infile.read())
        config = ModelConfig()
        config.update_from_dict(config_dict)
        return config
    
    
    def __init__(self):
        self.update_from_dict(default_values_dict, check_for_existing_key=False)
        
        
    def update_from_dict(self, new_vals, check_for_existing_key=False):
        for key, val in new_vals.items():
            assert type(key) == str
            if check_for_existing_key:
                assert key in self.__dict__, key
            self.__dict__[key] = val
        
        
    def as_dict(self):
        for i, column in enumerate(self.preprocess_encode_columns):
            self.preprocess_encode_columns[i] = column.__dict__
        return dict(self.__dict__)
    
    
    def get_copy(self):
        return deepcopy(self)
            
            
    def __repr__(self):
        return str(self)
            
            
    def __str__(self):
        return str(self.__dict__)
