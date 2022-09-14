#!/usr/bin/env python3
"""
Given one json filepath, creates a ModelManager instance and does predictions on the test and predict RecContexts.

Should take about 2.181 seconds per RecContext, for a total time of 10 hours.
HOWEVER, loading some pickle files is occasionally super slow, e.g.
2022-06-07 21:32:19.949 WARNING evaluateModelFromCache - evaluate_test_contexts: Processed RecContext 4049 in 0:00:05.167984; loading 0:00:04.802708, scoring 0:00:00.365276.
2022-06-07 21:33:07.149 WARNING evaluateModelFromCache - evaluate_test_contexts: Processed RecContext 4063 in 0:00:27.077212; loading 0:00:26.558092, scoring 0:00:00.519120.
2022-06-07 21:35:04.583 WARNING evaluateModelFromCache - evaluate_test_contexts: Processed RecContext 4104 in 0:00:56.526879; loading 0:00:56.163508, scoring 0:00:00.363371.
2022-06-07 21:36:27.005 WARNING evaluateModelFromCache - evaluate_test_contexts: Processed RecContext 4159 in 0:00:06.322407; loading 0:00:04.826389, scoring 0:00:01.496018.

TODO consider writing this with multiple worker threads that read in the RecContexts, since those might be able to do simultaneous IO / pass on the loaded recContexts for eval as soon as they are loaded (not respecting the order)
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
    sys.path.append("/home/lana/levon003/repos/recsys-peer-match/src")

import cbrec.featuredb
import cbrec.genconfig
import cbrec.utils
import cbrec.modeling.text_loader
import cbrec.modeling.reccontext_builder
import cbrec.modeling.scorer
import cbrec.modeling.manager

GLOBAL_SCRATCH_DIR = "/scratch.global/levon003/rec_context_md/"

VALIDATION_END_TIMESTAMP = datetime.strptime("2021-07-01", "%Y-%m-%d").timestamp() * 1000

def evaluate_test_contexts(config, model_filepath, evaluation_type):
    """
    :evaluation_type - 'eval' (meaning both), 'validation', or 'test'
    """
    if evaluation_type not in ['eval', 'validation', 'test']:
        raise ValueError("Unknown evaluation_type.")
    logger = logging.getLogger("cbrec.modeling.evaluateModel.evaluate_test_contexts")
    logger.info(f"Loading model from '{model_filepath}'.")
    
    manager = cbrec.modeling.manager.ModelManager.load_from_filepath(model_filepath)
    manager.load_model()
    
    metadata_filepath = os.path.join(manager.model_config.output_dir, f'{manager.model_config.experiment_name}_{manager.model_config.output_name}_{evaluation_type}_metadata.ndjson')
    logger.info(f"Writing metrics to '{metadata_filepath}'.")
    
    test_md_list = [md 
                    for md in cbrec.utils.stream_metadata_list(config.metadata_filepath) 
                    if md['type'] == 'test' 
                        and (evaluation_type == 'both' 
                             or (evaluation_type == 'validation' and md['timestamp'] <= VALIDATION_END_TIMESTAMP)
                             or (evaluation_type == 'test' and md['timestamp'] > VALIDATION_END_TIMESTAMP)
                            )
    ]
    logger.info(f"{len(test_md_list)} test metadata entries loaded (evaluation_type = {evaluation_type}).")
    db = cbrec.featuredb.get_db_by_filepath(config.feature_db_filepath)
    rec_md_dir = GLOBAL_SCRATCH_DIR
    with db, open(metadata_filepath, 'w') as metadata_outfile:
        for i in tqdm(range(len(test_md_list)), desc='Loading RecContexts'):
            filepath = os.path.join(rec_md_dir,  f"reccontext{i}.pkl")
            started_at = datetime.now()
            rc = None
            with open(filepath,'rb') as file:
                rc = pickle.load(file)
            loaded_at = datetime.now()
            # TODO if rc.has_target is False, then should save the scores separately. Use the approach we're using for the baseline models.
            scorer = manager.score_reccontext(rc)
            scored_at = datetime.now()
            metadata_outfile.write(json.dumps(rc.md) + '\n')
            if (scored_at - started_at).total_seconds() >= 5:
                logger.warning(f"Processed RecContext {i} in {scored_at - started_at}; loading {loaded_at - started_at}, scoring {scored_at - loaded_at}.")
    logger.info("Finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-filepath', dest='model_filepath', required=True)
    parser.add_argument('--validation-only', dest='validation_only', action='store_true')
    parser.add_argument('--test-only', dest='test_only', action='store_true')
    args = parser.parse_args()
        
    config = cbrec.genconfig.Config()
    
    if args.validation_only and args.test_only:
        raise ValueError("Only one of --validation-only and --test-only can be specified.")
    evaluation_type = 'eval'
    if args.validation_only:
        evaluation_type = 'validation'
    elif args.test_only:
        evaluation_type = 'test'
    
    evaluate_test_contexts(config, args.model_filepath, evaluation_type)
    

if __name__ == "__main__":
    main()
