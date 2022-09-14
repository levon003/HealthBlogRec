
import os
import json
import logging
import numpy as np
from glob import glob
from datetime import datetime
import sklearn.preprocessing

import cbrec.genconfig
import cbrec.modeling.modelconfig
import cbrec.modeling.train
import cbrec.modeling.preprocess
import cbrec.modeling.scorer


class ModelManager:
    """
    Manages models.
    
    Note: does not actually store the model itself, but offers a variety of convenient functions to manage I/O and interface with a ModelTrainer.
    """
    def load_from_output_name(output_name, output_dir=None):
        """
        Note: untested. Generally prefer the load_from_model_name function.
        """
        logger = logging.getLogger("cbrec.modeling.manager.ModelManager.load_from_output_name")
        if output_dir is None:
            output_dir = cbrec.modeling.modelconfig.ModelConfig().output_dir
        if output_name.endswith(".json"):  # if the given output name already ends with .json, strip it
            output_name = output_name[:-5]
        
        model_json_filepath = os.path.join(output_dir, output_name + ".json")
        if not os.path.exists(model_json_filepath):
            raise ValueError(f"Expected to find model with {output_name} at '{model_json_filepath}'; file not found.")
        model_manager = ModelManager.load_from_filepath(model_json_filepath)
        if model_manager.model_config.output_name != output_name:
            logger.warning(f"Using {output_name}, but loaded a config with output_name={model_config.output_name}.")
        return model_manager
    
    
    def load_from_model_name(model_name, experiment_name='default', output_dir=None):
        """
        Appropriate to use for loading model configurations saved with model_config.output_name == 'default'.
        """
        logger = logging.getLogger("cbrec.modeling.manager.ModelManager.load_from_model_name")
        if output_dir is None:
            output_dir = cbrec.modeling.modelconfig.ModelConfig().output_dir
        # default uses datetime
        # so, may be multiple options
        # if so, load the most recent
        candidate_pattern = os.path.join(output_dir, f"{model_name}_{experiment_name}_*.json")
        candidate_paths = glob(candidate_pattern)
        if len(candidate_paths) > 1:
            candidate_paths.sort(reverse=True)
            model_json_filepath = candidate_paths[0]
            logger.info(f"From {len(candidate_paths)} existing candidate paths, selected most recent path '{model_json_filepath}'.")
        elif len(candidate_paths) == 1:
            model_json_filepath = candidate_paths[0]
            logger.info(f"Loading from sole candidate path '{model_json_filepath}'.")
        else:
            raise ValueError(f"No path found matching {candidate_pattern}.")
        return ModelManager.load_from_filepath(model_json_filepath)
    
    
    def load_from_filepath(model_json_filepath):
        logger = logging.getLogger("cbrec.modeling.manager.ModelManager.load_from_filepath")
        with open(model_json_filepath, 'r') as infile:
            model_json = json.loads(infile.read())
        logger.info(f"Loaded existing model JSON output from '{model_json_filepath}'.")
        config_dict = model_json['config']
        model_config = cbrec.modeling.modelconfig.ModelConfig.from_dict(config_dict)
        
        old_output_basename = model_config.output_basename
        model_manager = ModelManager(model_config)
        if model_config.output_basename != old_output_basename:
            logger.info(f"Overwriting new output basename {model_config.output_basename} with existing output basename {old_output_basename}.")
            model_config.output_basename = old_output_basename
            
        model_manager.model_trainer.best_model_description = model_json['best_model_description']
        
        return model_manager
    
    
    def __init__(self,
                    model_config: cbrec.modeling.modelconfig.ModelConfig,
                    config: cbrec.genconfig.Config=None,
                ):
        self.config = config
        if self.config is None:
            # we don't require a genconfig to be passed, since it is only used by the parent Scoring implementation
            self.config = cbrec.genconfig.Config()
        self.model_config = model_config
        self.logger = logging.getLogger("cbrec.modeling.manager.ModelManager")
        
        # set output_basename, which is the name used when saving data associated with this model
        if self.model_config.output_name == 'default':
            self.model_config.output_basename = f"{self.model_config.model_name}_{self.model_config.experiment_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        else:
            self.model_config.output_basename = self.model_config.output_name
        
        self.feature_manager = cbrec.modeling.preprocess.FeatureManager(self.model_config)
        self.model_trainer = cbrec.modeling.train.ModelTrainer(self.model_config, self.feature_manager)
            
    
    def train_model(self, X, y_true):
        """
        Train a model using the current configuration.
        
        :X - the training data matrix. By convention, this matrix is expected to have 1563 columns.
        :y_true - the training data true labels.  By convention, each pair of adjacent rows is considered to correspondend to a source/target and source/alt pairing with targets 1 and 0 respectively.
        """
        
        # create a train/validation split
        # TODO create a new default_rng instance and randomize the subset of data? right now we choose the data temporally to be the last 1% by date iirc
        n_train = int(np.ceil(len(y_true) * 0.99))
        X_train = X[:n_train,:]
        X_test = X[n_train:,:]
        y_train = y_true[:n_train]
        y_test = y_true[n_train:]
        self.logger.info(f"Train/validation sizes: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")
        
        # apply data transforms
        X_train = self.feature_manager.fit_transform(X_train)
        X_test = self.feature_manager.transform(X_test)
        
        self.model_trainer.train_model(X_train, y_train, X_test, y_test)
        self.logger.info("Finished training net.")
        
        
    def score_test_matrix(self, X_test):
        X_test = self.feature_manager.transform(X_test)
        y_test_score = self.model_trainer.score_matrix(X_test)
        return y_test_score
    
    
    def score_reccontext(self, rc, update_md=True):
        scorer = cbrec.modeling.scorer.ModelScorer(self.config, rc, self, save_scores=True)
        metrics = scorer.score()
        if update_md:
            rc.md[f'{self.model_config.output_name}_metrics'] = metrics
        return scorer
        
        
    def load_model(self, load_preprocessor=True, load_model_state_dict=True, load_training_metrics=False, use_best_model=True):
        logger = logging.getLogger("cbrec.modeling.manager.ModelManager.load_model")
        
        if load_preprocessor:
            self.feature_manager.load_learned_transformations()
            logger.info("Loaded the preprocessor's learned transformations.")
        
        if load_model_state_dict:
            if use_best_model:
                if self.model_trainer.best_model_description is None:
                    logger.warn(f"Requested a model state dict load with the best model, but no best model description available.")
            self.model_trainer.load_model_state_dict(description=self.model_trainer.best_model_description)
            logger.info("Loaded model state dict.")
            
        if load_training_metrics:
            self.model_trainer.load_train_metrics()
            logger.info("Loaded train metrics.")

    
    def save_model(self, save_preprocessor=True, save_model_state_dict=True, save_training_metrics=True):
        logger = logging.getLogger("cbrec.modeling.manager.ModelManager.save_model")
        
        if not os.path.exists(self.model_config.output_dir):
            os.makedirs(self.model_config.output_dir)
        
        if save_preprocessor:
            self.feature_manager.save_learned_transformations()
        
        state_dict_filepath = None
        if save_model_state_dict:
            state_dict_filepath = self.model_trainer.save_model_state_dict()

        train_metrics_filepath, valid_metrics_filepath = None, None
        if save_training_metrics:
            train_metrics_filepath, valid_metrics_filepath = self.model_trainer.save_train_metrics()
            
        # in the future, also save any automated evaluation procedure that happens after training
        
        output_json = {
            'config': self.model_config.as_dict(),
            'output_dir': self.model_config.output_dir,
            'output_basename': self.model_config.output_basename,
            'state_dict_filepath': state_dict_filepath,
            'train_metrics_filepath': train_metrics_filepath,
            'valid_metrics_filepath': valid_metrics_filepath,
            'best_model_description': self.model_trainer.best_model_description,
        }
        model_json_filepath = os.path.join(self.model_config.output_dir, self.model_config.output_basename + ".json")
        with open(model_json_filepath, 'w') as outfile:
            outfile.write(json.dumps(output_json) + "\n")
        logger.info(f"Wrote model outputs using basename '{self.model_config.output_basename}' to '{model_json_filepath}'.")