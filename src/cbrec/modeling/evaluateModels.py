#!/usr/bin/env python3
# Script to evaluate models
"""
Given one or more ModelManager instances, does predictions on the test set.

Steps:
Create text cache (which loads database contents into memory)
Stream test contexts from the database
Create RecContext instance
Createi X_test in the RecContext
    Create a new class that uses the text cache to create a matrix efficiently
    In other words, write a better version of FeatureLoader.combine_feature_arrs()?
for mm in model_managers:
    scorer = ModelManager.score_reccontext()
-Do something with scorer.metrics_dict


"""

import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

import sys
import os
import json
import itertools
from glob import glob
from tqdm import tqdm
from datetime import datetime
#import sqlite3
#import multiprocessing as mp
#import threading
import argparse
import numpy as np
import torch
import pickle
try:
    import cbrec
except:
    sys.path.append("/home/lana/nguy4068/repos/recsys-peer-match/src")

import cbrec.featuredb
import cbrec.genconfig
import cbrec.utils
import cbrec.modeling.text_loader
import cbrec.modeling.reccontext_builder
import cbrec.modeling.scorer
import cbrec.modeling.manager


def evaluate_test_contexts(config, model_filepaths):
    logger = logging.getLogger("cbrec.modeling.evaluateModels.evaluate_test_contexts")
    
    managers = []
    for model_filepath in model_filepaths:
        manager = cbrec.modeling.manager.ModelManager.load_from_filepath(model_filepath)
        manager.load_model()
        managers.append(manager)
    if len(set([manager.model_config.experiment_name for manager in managers])) != 1:
        raise ValueError(f"Managers from multiple experiments are currently not supported. Let Zach know if you need this feature.")
    metadata_filepath = os.path.join(managers[0].model_config.output_dir, f'{managers[0].model_config.experiment_name}_test_metadata.ndjson')
    logger.info(f"Writing metrics to '{metadata_filepath}'.")
    
    text_loader = cbrec.modeling.text_loader.TextLoader(config)
    text_loader.cache_all_journals()
    logger.info("TextLoader created and embeddings cached.")
    
    test_md_list = [md for md in cbrec.utils.stream_metadata_list(config.metadata_filepath) if md['type'] == 'test']
    logger.info(f"{len(test_md_list)} test metadata entries loaded.")
    test_md_map = {md['metadata_id']: md for md in test_md_list}
    
    db = cbrec.featuredb.get_db_by_filepath(config.feature_db_filepath)
    with db, open(metadata_filepath, 'w') as metadata_outfile:
        for test_context in tqdm(cbrec.featuredb.stream_test_contexts(db, config), desc='Streaming test contexts', total=len(test_md_map)):
            test_context_md = test_md_map[test_context['metadata_id']]
            rc = cbrec.modeling.reccontext_builder.build_reccontext(config, text_loader, test_context_md, test_context)
            for manager in managers:
                scorer = manager.score_reccontext(rc)
            #metrics = scorer.metrics_dict[scorer.model_name]
            #metrics_outfile.write(json.dumps(metrics) + '\n')
            # the scorer updates the RecContext's metadata (md) entry, so we save that to a file
            metadata_outfile.write(json.dumps(rc.md) + '\n')
    
    logger.info("Finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-filepath', dest='model_filepath', required=False)
    parser.add_argument('--model-dir', dest='model_dir', required=False)
    args = parser.parse_args()
        
    config = cbrec.genconfig.Config()
    
    if args.model_filepath is not None:
        model_filepaths = [args.model_filepath,]
        if not os.path.exists(args.model_filepath):
            raise ValueError(f"Filepath '{model_filepath}' does not exist.")
    elif args.model_dir is not None:
        model_filepaths = []
        if not os.path.exists(args.model_dir):
            raise ValueError(f"Dir '{args.model_dir}' does not exist.")
        for model_filepath in glob(os.path.join(args.model_dir, '*.json')):
            model_filepaths.append(model_filepath)
        if len(model_filepaths) == 0:
            raise ValueError(f"No .json files in dir '{args.model_dir}'.")
        logging.info(f"Identified {len(model_filepaths)} model filepaths in dir {args.model_dir}.")
    else:
        raise ValueError("One of --model-filepath or --model-dir must be specified.")

    evaluate_test_contexts(config, model_filepaths)
    

if __name__ == "__main__":
    main()
