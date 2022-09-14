#!/usr/bin/env python

# Script to generate a model, train it, and save its results. Commands can be passed into
# the script to change the repository file path and the model parameters.

# First command: personal repository name (i.e. mcnam385 or levon003).
#    This ensures that the script will run with the code in your own respository. If no
#    command is passed in, then 'mcnam385' will be used by default
# Second command: json config file name. This file must be stored in the experiment/jsons directory.
#    If no command is passed in, then the default config file will be used by default.

# Example scripts:
# $ python model_gen.py
# $ python model_gen.py mcnam385
# $ python model_gen.py levon003 example_config.json

import numpy as np
import os
import re
import json
import sys
import pickle
from tqdm import tqdm
import sklearn
import sklearn.preprocessing
import dateutil.parser
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import pytz
import torch


def main():
    # setting file path for personal repo. levon003 is the default
    if len(sys.argv) < 2:
        username = 'levon003'
    else:
        username = sys.argv[1]
    sys.path.append(os.path.join('/home/lana/' + username + '/repos/recsys-peer-match', 'src'))
    import cbrec.genconfig
    import cbrec.evaluation
    import cbrec.torchmodel
    import cbrec.logutils
    import cbrec.modeling.modelconfig
    import cbrec.modeling.scorer
    import cbrec.modeling.manager

    # setting up logging
    cbrec.logutils.set_up_logging()
    import logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logger = logging.getLogger("cbrec.modeling.trainModel.main")

    # setting file path to caringbridge core
    caringbridge_core_path = "/home/lana/levon003/repos/caringbridge_core"
    sys.path.append(caringbridge_core_path)
    import cbcore.data.paths

    # loading training features
    feature_cache_dir = os.path.join(cbcore.data.paths.projects_data_dir, 'recsys-peer-match', 'torch_experiments', 'feature_cache')
    filenames = [
        ('X_train_raw.pkl', 'y_train_raw.pkl'),
#        ('X_test2train_raw.pkl', 'y_test2train_raw.pkl'),
    ]
    Xs = []
    ys = []
    for x_filename, y_filename in filenames:
        with open(os.path.join(feature_cache_dir, x_filename), 'rb') as infile:
            X = pickle.load(infile)
            Xs.append(X)
        with open(os.path.join(feature_cache_dir, y_filename), 'rb') as infile:
            y = pickle.load(infile)
            ys.append(y)
    X = np.concatenate(Xs, axis=0)
    y_true = np.concatenate(ys, axis=0)

    # shuffle the data
    inds = np.arange(len(X))
    np.random.shuffle(inds)
    X = X[inds]
    y_true = y_true[inds]

    if len(sys.argv) < 3:
        # generates default cbrec model
        model_config = cbrec.modeling.modelconfig.ModelConfig()
    else:
        model_config = cbrec.modeling.modelconfig.ModelConfig.from_filepath(sys.argv[2])
        
    model_manager = cbrec.modeling.manager.ModelManager(model_config)
    logger.info(model_manager.model_config.output_basename)

    model_manager.train_model(X, y_true)
    model_manager.save_model()

    logger.info("Finished training and saving model.")

if __name__ == '__main__':
    main()
