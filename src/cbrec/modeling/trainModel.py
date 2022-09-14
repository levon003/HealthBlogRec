#!/usr/bin/env python
# Script to train PyTorch models from the current cache
# temporary/hacky, but allows training outside a notebook

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

sys.path.append(os.path.join('/home/lana/levon003/repos/recsys-peer-match', 'src'))
import cbrec.genconfig
import cbrec.evaluation
import cbrec.torchmodel
import cbrec.logutils

cbrec.logutils.set_up_logging()
# turn off matplotlib logging
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger("cbrec.modeling.trainModel.main")

caringbridge_core_path = "/home/lana/levon003/repos/caringbridge_core"
sys.path.append(caringbridge_core_path)
import cbcore.data.paths

# load train features
feature_cache_dir = os.path.join(cbcore.data.paths.projects_data_dir, 'recsys-peer-match', 'torch_experiments', 'feature_cache')
filenames = [
    ('X_train_raw.pkl', 'y_train_raw.pkl'),
    ('X_test2train_raw.pkl', 'y_test2train_raw.pkl'),
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

import cbrec.modeling.modelconfig
import cbrec.modeling.scorer
import cbrec.modeling.manager

model_config = cbrec.modeling.modelconfig.ModelConfig()
model_config.experiment_name = 'main'
model_config.train_n_epochs = 1200
model_config.train_weight_decay = 0.0001
#model_config.train_max_lr = 0.013
model_config.LinearNet_dropout_p = 0.5

model_manager = cbrec.modeling.manager.ModelManager(model_config)
logger.info(model_manager.model_config.output_basename)

model_manager.train_model(X, y_true)
model_manager.save_model()

logger.info("Finished training and saving model")

