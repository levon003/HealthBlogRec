#!/usr/bin/env python

# Script to generate all the config files for a given experiment.
# Additionally, the script generates a run_all.sh file to train and evaluate each config file and store the results.

# This script takes in two commands:
# First command: username (e.g. mcnam385 or levon003)
# Second command: name of experiment

# In order to implement different experiments, a new function must be created. See example implementations.

import sys
import os
import json
import stat
import logging
import argparse
from datetime import datetime, timedelta
import scipy
from scipy.stats import loguniform, uniform, lognorm, norm
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--username', dest='username', required=True)
parser.add_argument('--experiment-name', dest='experiment_name', required=True)
args = parser.parse_args()

src_path = f'/home/lana/{args.username}/repos/recsys-peer-match/src'  # hacky, but fine for now
assert os.path.exists(src_path), f"Expected directory '{src_path}' does not exist."
sys.path.append(src_path)

import cbrec.modeling.modelconfig
import cbrec.logutils

class Experiment:
    """
    To create an experiment:
    -Create a new Experiment() with the x500 username of the invoking user and a name for the experiment.
    -Create a root ModelConfig (the default or some variant)
    -Create as many configs as you want. For each:
        Set output_name to a unique name (including Experiment.config_counter in the name ensures uniqueness)
        Set output_dir to Experiment.output_dir
        Call Experiment.save_config
    -Call Experiment.finalize_scripts()
    -Call Experiment.log_instructions()
    
    See the example usages in this file.

    """
    def __init__(self, username, experiment_name):
        self.logger = logging.getLogger('cbrec.experiment.Experiment')
        
        self.username = username
        self.experiment_name = experiment_name
        
        self.config_counter = 0
        self.configs = []
        
        self.src_path = f'/home/lana/{self.username}/repos/recsys-peer-match/src'
        if not os.path.exists(self.src_path):
            self.logger.warning(f"Dir '{self.src_path}' does not exist. Continuing; but check configuration.")
        
        self._create_dirs()
        
    def _create_dirs(self):
        # build directory structure for this experiment
        # experiment (main directory), configs (stores config files), outputs (saves model evaluation data), sbatch (output and errors for Slurm scripts)
        model_dir = '/home/lana/shared/caringbridge/data/projects/recsys-peer-match/torch_experiments/modeling'
        assert os.path.exists(model_dir)
        self.experiment_dir = f'{model_dir}/{self.experiment_name}_{datetime.utcnow().strftime("%Y%m%d%H%M%S")}'
        self.config_dir = f'{self.experiment_dir}/configs'
        self.scripts_dir = f'{self.experiment_dir}/scripts'
        self.output_dir = f'{self.experiment_dir}/outputs'
        self.sbatch_dir = f'{self.experiment_dir}/sbatch'
        os.mkdir(self.experiment_dir)
        os.mkdir(self.config_dir)
        os.mkdir(self.scripts_dir)
        os.mkdir(self.output_dir)
        os.mkdir(self.sbatch_dir)
        self.logger.info(f"Created experiment dirs at path '{self.experiment_dir}'.")
        
        # define path for scripts
        self.submit_script_filepath = f'{self.scripts_dir}/submit_train_all.sh'
        self.train_script_filepath = f'{self.scripts_dir}/run_train_all.sh'
        self.eval_script_filepath = f'{self.scripts_dir}/eval_all.sh'
        
    def save_config(self, config: cbrec.modeling.modelconfig.ModelConfig, config_name=None):
        """
        Note: config_name can be overridden, but are you sure you want to do that? Easiest to keep filenames == config.output_name, which encourages keeping config.output_name unique in an experiment.
        """
        if config_name is None:
            config_name = config.output_name
        if config_name == 'default':
            self.logger.warning(f"Generated a config with output_name 'default'. Continuing, but verify configuration.")
        if config.experiment_name != self.experiment_name:
            self.logger.warning(f"Trying to generate a config with experiment_name '{config.experiment_name}'. Overriding with '{self.experiment_name}', but verify configuration.")
            config.experiment_name = self.experiment_name
        config_filepath = f'{self.config_dir}/{config_name}.json'
        if os.path.exists(config_filepath):
            self.logger.warning(f"Unexpected duplicate JSON at path '{config_filepath}'. Continuing, but verify configuration.")
        with open(config_filepath, 'w') as config_json_outfile:
            json.dump(config.as_dict(), config_json_outfile)
        with open(self.submit_script_filepath, 'a') as script_outfile:
            script_outfile.write(f'sbatch -p agsmall --mail-type=FAIL --mail-user={self.username}@umn.edu --job-name={config_name} --output={self.sbatch_dir}/{config_name}.stdout --error={self.sbatch_dir}/{config_name}.stderr --export=USERNAME=\'{self.username}\',CONFIG=\'{config_filepath}\' {self.src_path}/cbrec/experiment/model_gen_script.sh\n')
        with open(self.train_script_filepath, 'a') as script_outfile:
            script_outfile.write(f'python {os.path.join(src_path, "cbrec/experiment/model_gen.py")} {self.username} {config_filepath}\n')
        self.config_counter += 1
        self.configs.append(config)
        
    def log_instructions(self):
        instructions = """Config generation complete!
To submit models via sbatch:
{submit_script_filepath}
To train models directly:
{train_script_filepath}

To submit eval via sbatch (after training has finished, newer approach):
python cbrec/modeling/submitEvalFromDirectory.py --model-dir {output_dir} --validation-only

To submit eval via sbatch (after training, old approach, probably don't use):
sbatch -p agsmall {eval_script_filepath}
    
Upon the script's completion, all model data can be found here:
{experiment_dir}
        """.format(
            submit_script_filepath=self.submit_script_filepath, 
            train_script_filepath=self.train_script_filepath, 
            output_dir=self.output_dir,
            eval_script_filepath=self.eval_script_filepath, 
            experiment_dir=self.experiment_dir,
        )
        
        self.logger.info(instructions)
    
    def finalize_scripts(self):
        self._create_eval_script()
        
        with open(self.submit_script_filepath, 'a') as script_outfile:
            script_outfile.write(f'echo "Finished submitting."\n')
        
        # ug+rwx
        exec_permissions = stat.S_IRUSR|stat.S_IWUSR|stat.S_IXUSR|stat.S_IRGRP|stat.S_IWGRP|stat.S_IXGRP
        os.chmod(self.submit_script_filepath, exec_permissions)
        os.chmod(self.train_script_filepath, exec_permissions)
        os.chmod(self.eval_script_filepath, exec_permissions)
    
    def _create_eval_script(self):
        eval_script="""#!/bin/bash -l        
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=12g
#SBATCH --mail-type=ALL
#SBATCH --mail-user={username}@umn.edu
#SBATCH --job-name=cb_recsys_evaluateModels
#SBATCH -o {sbatch_dir}/evaluateModels.stdout
#SBATCH -e {sbatch_dir}/evaluateModels.stderr
working_dir="{src_path}"
cd $working_dir
source activate pytorch-cpuonly || exit 1
echo "In '$working_dir', running script."
python cbrec/modeling/evaluateModels.py --model-dir {output_dir}
echo "Finished sbatch script."
""".format(
            username=self.username,
            src_path=self.src_path,
            sbatch_dir=self.sbatch_dir,
            output_dir=self.output_dir,
        )
        with open(self.eval_script_filepath, 'w') as script_outfile:
            script_outfile.write(eval_script)


def learning_rate_experiment(username):
    experiment_name = 'learning_rate_experiment'
    e = Experiment(username, experiment_name)
    
    root_config = cbrec.modeling.modelconfig.ModelConfig()
    root_config.experiment_name = experiment_name
    root_config.output_dir = e.output_dir
    # can set other changes to defaults here
    
    # iterate through 10 different 'train_max_lr' values evenly spaced between 0.01 and 0.02
    for i in range(10):
        config = root_config.get_copy()
        config.output_name = f'{experiment_name}_{e.config_counter}'
        new_max_lr = 0.01 + i/1000
        config.train_max_lr = new_max_lr
        e.save_config(config)
    
    e.finalize_scripts()
    e.log_instructions()
    
    
def field_study_model_experiment(username):
    """
    The basic LinearNet used during the study, fit with 3 different models for each of the conditions (4 hyperparameters).
    """
    experiment_name = 'field_study_model_experiment'
    e = Experiment(username, experiment_name)
    
    root_config = cbrec.modeling.modelconfig.ModelConfig()
    root_config.experiment_name = experiment_name
    root_config.output_dir = e.output_dir
    # set other changes to defaults here
    root_config.lr_init = 0.01
    #train_max_lr': 0.0155,
    
    for train_max_lr in [0.008, 0.01, 0.012, 0.014, 0.016]:
        for LinearNet_n_hidden in [100, 300, 500]:
            for train_weight_decay in [0, 0.0001, 0.01]:
                for LinearNet_dropout_p in [0, 0.1, 0.5, 0.9]:
                    for i in range(3):
                        config = root_config.get_copy()
                        config.output_name = f'{experiment_name}_{e.config_counter}'
                        config.train_max_lr = train_max_lr
                        config.LinearNet_n_hidden = LinearNet_n_hidden
                        config.train_weight_decay = train_weight_decay
                        config.LinearNet_dropout_p = LinearNet_dropout_p
                        e.save_config(config)
    
    e.finalize_scripts()
    e.log_instructions()
    
    
def LinearNet_feature_ablations(username):
    """
    Feature ablations, with hyperparameters as set by the field_study_model_experiment
    """
    experiment_name = 'LinearNet_feature_ablations'
    e = Experiment(username, experiment_name)
    
    root_config = cbrec.modeling.modelconfig.ModelConfig()
    root_config.experiment_name = experiment_name
    root_config.output_dir = e.output_dir
    # set other changes to defaults here
    root_config.lr_init = 0.01
    root_config.train_weight_decay = 0.0001
    root_config.LinearNet_dropout_p = 0.5
    
    for output_name, preprocess_drop_columns in [
#        ('all', []),  # NOT including the 'all' condition, since it's redundant with the results in field_study_model_experiment
        ('text_only', [['*', '~text']]),
        ('network_only', [['*', '~network']]),
        ('activity_only', [['*', '~activity']]),
        ('nontext_only', [['*', 'text']]),
        ('nonnetwork_only', [['*', 'network']]),
        ('nonactivity_only', [['*', 'activity']]),
    ]:
        for train_max_lr in [0.008, 0.01, 0.012, 0.014, 0.016]:
            for LinearNet_n_hidden in [10, 100, 300]:
                for i in range(3):
                    config = root_config.get_copy()
                    config.output_name = f'{experiment_name}_{output_name}_{e.config_counter}'
                    config.train_max_lr = train_max_lr
                    config.LinearNet_n_hidden = LinearNet_n_hidden
                    config.preprocess_drop_columns = preprocess_drop_columns
                    e.save_config(config)
    
    e.finalize_scripts()
    e.log_instructions()

    
def adam_randomsearch_experiment(username):
    experiment_name = 'adam_randomsearch_experiment'
    e = Experiment(username, experiment_name)
    
    root_config = cbrec.modeling.modelconfig.ModelConfig()
    root_config.experiment_name = experiment_name
    root_config.output_dir = e.output_dir
    root_config.train_scheduler_name = 'None'
    root_config.train_weight_decay = 0
    
    # create scipy distributions with the appropriate parameters
    budget_size = 100
    rv = lognorm(s=1.42, scale=np.exp(-2.69))
    lr_values = rv.rvs(size=budget_size)
    rv = loguniform(np.exp(-5), np.exp(-1))
    b1_values = 1 - rv.rvs(size=budget_size)
    b2_values = 1 - rv.rvs(size=budget_size)
    rv = loguniform(np.exp(-8), np.exp(0))
    e_values = rv.rvs(size=budget_size)
    
    for i in range(budget_size):
        config = root_config.get_copy()
        config.output_name = f'{experiment_name}_{e.config_counter}'
        # set Adam parameters by sampling from the distributions
        config.train_lr_init = lr_values[i]
        config.train_Adam_beta1 = b1_values[i]
        config.train_Adam_beta2 = b2_values[i]
        config.train_Adam_eps = e_values[i]
        e.save_config(config)
    
    e.finalize_scripts()
    e.log_instructions()
    

def simnet_experiment(username):
    experiment_name = 'simnet'
    e = Experiment(username, experiment_name)
    
    root_config = cbrec.modeling.modelconfig.ModelConfig()
    root_config.experiment_name = experiment_name
    root_config.model_name = 'SimNet'
    root_config.output_dir = e.output_dir
    root_config.train_n_epochs = 0
    root_config.preprocess_use_scaler = True
    #root_config.preprocess_drop_columns = ['shared-are_weakly_connected', 'shared-is_fof', 'shared-is_reciprocal']
    
    for output_name, preprocess_drop_columns in [
        ('all', [['shared', '*']]),
        ('text_only', [['*', '~text']]),
        ('nontext_only', [['shared', '*'], ['*', 'text']]),
    ]:
        config = root_config.get_copy()
        config.output_name = f'{experiment_name}_{output_name}'
        config.preprocess_drop_columns = preprocess_drop_columns
        e.save_config(config)
    
    e.finalize_scripts()
    e.log_instructions()
    
    
def concatnet_experiment(username):
    experiment_name = 'concatnet'
    e = Experiment(username, experiment_name)
    
    root_config = cbrec.modeling.modelconfig.ModelConfig()
    root_config.experiment_name = experiment_name
    root_config.model_name = 'ConcatNet'
    root_config.output_dir = e.output_dir
    #root_config.preprocess_drop_columns = ['shared-are_weakly_connected', 'shared-is_fof', 'shared-is_reciprocal']
    root_config.train_n_epochs = 1000
    learning_rates = []
    for i in range(1,10):
        new_max_lr = i/10
        learning_rates.append(new_max_lr)
    for i in range(1,10):
        new_max_lr = 1 + i/10
        learning_rates.append(new_max_lr)
    for i in range(160,170):
        for j in learning_rates:
            output_name = f'hidden_layers_{i}_lr_{j}'
            config = root_config.get_copy()
            config.output_name = f'{experiment_name}_{output_name}'
            config.ConcatNet_n_hidden = i
            config.train_max_lr = j
            e.save_config(config)
    e.finalize_scripts()
    e.log_instructions()

def feature_ablations_experiment(username):
    experiment_name = 'feature_ablations'
    e = Experiment(username, experiment_name)
    
    root_config = cbrec.modeling.modelconfig.ModelConfig()
    root_config.experiment_name = experiment_name
    root_config.model_name = 'LinearNet'
    root_config.output_dir = e.output_dir
    root_config.train_scheduler_name = 'None'
    root_config.train_weight_decay = 0
    root_config.train_n_epochs = 1000
    #root_config.preprocess_drop_columns = ['shared-are_weakly_connected', 'shared-is_fof', 'shared-is_reciprocal']
    
    # create 150 Adam hyperparameter configurations
    num_adam_configs = 150
    rv = lognorm(s=1.42, scale=np.exp(-2.69))
    lr_values = rv.rvs(size=num_adam_configs)
    rv = loguniform(np.exp(-5), np.exp(-1))
    b1_values = 1 - rv.rvs(size=num_adam_configs)
    b2_values = 1 - rv.rvs(size=num_adam_configs)
    rv = loguniform(np.exp(-8), np.exp(0))
    e_values = rv.rvs(size=num_adam_configs)
    
    for output_name, preprocess_drop_columns in [
        ('all', []),
        ('text_only', [['*', '~text']]),
        ('network_only', [['*', '~network']]),
        ('activity_only', [['*', '~activity']]),
        ('nontext_only', [['*', 'text']]),
        ('nonnetwork_only', [['*', 'network']]),
        ('nonactivity_only', [['*', 'activity']]),
    ]:
    
        # build config with unique Adam hyperparameters and feature ablations
        for i in range(num_adam_configs):
            config = root_config.get_copy()
            config.output_name = f'{experiment_name}_{output_name}_{i+1}'
            
            # set drop columns
            config.preprocess_drop_columns = preprocess_drop_columns
            
            # set Adam hyperparameters
            config.train_lr_init = lr_values[i]
            config.train_Adam_beta1 = b1_values[i]
            config.train_Adam_beta2 = b2_values[i]
            config.train_Adam_eps = e_values[i]
            
            e.save_config(config)
    
    e.finalize_scripts()
    e.log_instructions()

    
def feature_test_experiment(username):
    experiment_name = 'feature_test'
    e = Experiment(username, experiment_name)
    
    root_config = cbrec.modeling.modelconfig.ModelConfig()
    root_config.experiment_name = experiment_name
    root_config.model_name = 'LinearNet'
    root_config.output_dir = e.output_dir
    root_config.train_scheduler_name = 'None'
    root_config.train_weight_decay = 0
    root_config.train_n_epochs = 1
    
    for output_name, preprocess_drop_columns in [
        ('all', []),
        ('text_only', [['*', '~text']]),
        ('network_only', [['*', '~network']]),
        ('activity_only', [['*', '~activity']]),
        ('nontext_only', [['*', 'text']]),
        ('nonnetwork_only', [['*', 'network']]),
        ('nonactivity_only', [['*', 'activity']]),
    ]:
        config = root_config.get_copy()
        config.output_name = f'{experiment_name}_{output_name}'
        # set drop columns
        config.preprocess_drop_columns = preprocess_drop_columns
        # save the configs
        e.save_config(config)
    
    e.finalize_scripts()
    e.log_instructions()


def main():
    cbrec.logutils.set_up_logging()
    
    username, experiment_name = args.username, args.experiment_name
    
    if experiment_name == 'learning_rate_experiment':
        learning_rate_experiment(username)
    elif experiment_name == 'field_study_model_experiment':
        field_study_model_experiment(username)
    elif experiment_name == 'LinearNet_feature_ablations':
        LinearNet_feature_ablations(username)
    elif experiment_name == 'adam_randomsearch_experiment':
        adam_randomsearch_experiment(username)
    elif experiment_name == 'simnet_experiment':
        simnet_experiment(username)
    elif experiment_name == 'feature_ablations_experiment':
        feature_ablations_experiment(username)
    elif experiment_name == 'feature_test_experiment':
        feature_test_experiment(username)
    elif experiment_name == "concatnet_experiment":
        concatnet_experiment(username)
    else:
        raise ValueError(f"Unrecognized experiment name '{experiment_name}'.")
    
if __name__ == '__main__':
    main()
