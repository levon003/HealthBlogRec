#!/usr/bin/env python3
"""
Given one json filepath, creates a ModelManager instance and does predictions on the test and predict RecContexts.

Should take about 2.181 seconds per RecContext, for a total time of 10 hours, or 20 for everything.
HOWEVER, loading some pickle files is occasionally super slow, e.g.
2022-06-07 21:32:19.949 WARNING evaluateModelFromCache - evaluate_test_contexts: Processed RecContext 4049 in 0:00:05.167984; loading 0:00:04.802708, scoring 0:00:00.365276.
2022-06-07 21:33:07.149 WARNING evaluateModelFromCache - evaluate_test_contexts: Processed RecContext 4063 in 0:00:27.077212; loading 0:00:26.558092, scoring 0:00:00.519120.
2022-06-07 21:35:04.583 WARNING evaluateModelFromCache - evaluate_test_contexts: Processed RecContext 4104 in 0:00:56.526879; loading 0:00:56.163508, scoring 0:00:00.363371.
2022-06-07 21:36:27.005 WARNING evaluateModelFromCache - evaluate_test_contexts: Processed RecContext 4159 in 0:00:06.322407; loading 0:00:04.826389, scoring 0:00:01.496018.

This multiprocessing version appears to do 1.31 seconds per RecContext instead, because it papers over that problem.
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
import multiprocessing as mp
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


def load_reccontext(filepath, queue):
    with open(filepath,'rb') as file:
        rc = pickle.load(file)
    queue.put(rc)


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
    
    scores_filepath = os.path.join(manager.model_config.output_dir, f'{manager.model_config.experiment_name}_{manager.model_config.output_name}_coverage_scores.pkl')
    logger.info(f"Writing scores (if available) to '{scores_filepath}'.")
    
    count = 0
    test_md_list = []
    metadata_id_set = set()
    for md in cbrec.utils.stream_metadata_list(config.metadata_filepath):
        if md['type'] == 'test' or md['type'] == 'predict':  # these are the things with associated test_context entries
            if evaluation_type == 'eval' \
                    or (evaluation_type == 'validation' and md['timestamp'] <= VALIDATION_END_TIMESTAMP and md['has_target']) \
                    or (evaluation_type == 'test' and (md['timestamp'] > VALIDATION_END_TIMESTAMP or not md['has_target'])):
                test_md_list.append(count)
                metadata_id_set.add(md['metadata_id'])
            count += 1
    
    logger.info(f"{len(test_md_list)} test metadata entries loaded (evaluation_type = {evaluation_type}). First reccontext at index {test_md_list[0]}, last at index {test_md_list[-1]}.")
    rec_md_dir = GLOBAL_SCRATCH_DIR
    scores = []
    with open(metadata_filepath, 'w') as metadata_outfile, mp.Pool(processes=5) as pool:
        mpmanager = mp.Manager()
        queue = mpmanager.Queue(maxsize=5)
        
        for i in tqdm(test_md_list, desc='Requesting RecContexts'):
            filepath = os.path.join(rec_md_dir,  f"reccontext{i}.pkl")
            if not os.path.exists(filepath):
                raise ValueError(f"Filepath '{filepath}' expected but missing.")
            pool.apply_async(load_reccontext, (filepath, queue))
            
        for i in tqdm(test_md_list, desc='Processing RecContexts'):
            rc = queue.get(block=True, timeout=120)
            assert rc.metadata_id in metadata_id_set, "Unexpected metadata_id."
            scorer = manager.score_reccontext(rc)
            
            # if this is a prediction target only, need to save the scores
            if not rc.has_target:
                assert scorer.save_scores and scorer.model_name in scorer.scores_dict
                rec_scores = scorer.scores_dict[scorer.model_name]
                scores.append({
                    'metadata_id': rc.metadata_id, 
                    'scores': rec_scores
                })
                if len(scores) >= 1000:
                    with open(scores_filepath, 'wb') as scores_outfile:
                        pickle.dump(scores, scores_outfile)
                    logging.info(f"Saved pickle with {len(scores)} scores.")
                    scores = []
                
            metadata_outfile.write(json.dumps(rc.md) + '\n')
            
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
