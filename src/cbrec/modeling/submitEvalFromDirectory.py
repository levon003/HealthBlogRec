#!/usr/bin/env python3
# Script to evaluate models
"""
Given a directory, submits sbatch jobs for each model in that directory.

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
import argparse
import numpy as np



def identify_model_filepaths(model_dir):
    logger = logging.getLogger("cbrec.modeling.submitEvalFromDirectory.identify_model_filepaths")
    if not os.path.exists(model_dir):
        raise ValueError(f"Dir '{model_dir}' does not exist.")
    model_filepaths = []
    for model_filepath in glob(os.path.join(model_dir, '*.json')):
        model_filepaths.append(model_filepath)
    if len(model_filepaths) == 0:
        raise ValueError(f"No .json files in dir '{model_dir}'.")
    logger.info(f"Identified {len(model_filepaths)} model filepaths in dir {model_dir}.")
    return model_filepaths


def submit_eval_for_model_filepath(model_filepath, username, validation_only, test_only, dry_run=False):
    logger = logging.getLogger("cbrec.modeling.submitEvalFromDirectory.submit_eval_for_model_filepath")
    script_path = f"/home/lana/{username}/repos/recsys-peer-match/src/cbrec/modeling/evaluateModel.sh"
    logging_dir = os.path.abspath(os.path.join(os.path.dirname(model_filepath), '..', 'sbatch'))
    assert os.path.exists(logging_dir)
    model_name = os.path.splitext(os.path.basename(model_filepath))[0]
    if test_only != '':
        stdout_filepath = os.path.join(logging_dir, f"eval_test_{model_name}.stdout")
        stderr_filepath = os.path.join(logging_dir, f"eval_test_{model_name}.stderr")
    else:
        stdout_filepath = os.path.join(logging_dir, f"eval_{model_name}.stdout")
        stderr_filepath = os.path.join(logging_dir, f"eval_{model_name}.stderr")
    command = f"sbatch -p agsmall --mail-type=FAIL --mail-user={username}@umn.edu -o {stdout_filepath} -e {stderr_filepath} --job-name evaluateModel_{model_name} --export=USERNAME='{username}',MODEL_FILEPATH='{model_filepath}',VALIDATION_ONLY='{validation_only}',TEST_ONLY='{test_only}' {script_path}"
    logger.info(command)
    if not dry_run:
        os.system(command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', dest='username', required=True)
    parser.add_argument('--model-dir', dest='model_dir', required=False)
    parser.add_argument('--model-filepath', dest='model_filepath', required=False)
    parser.add_argument('--validation-only', dest='validation_only', action='store_true', required=False)
    parser.add_argument('--test-only', dest='test_only', action='store_true', required=False)
    args = parser.parse_args()
    if args.model_dir is None and args.model_filepath is None:
        raise ValueError("One of --model-dir or --model-filepath must be specified.")
    
    validation_only = '--validation-only' if args.validation_only else ''
    test_only = '--test-only' if args.test_only else ''
    
    if args.model_filepath:
        logging.info(f"Submitting eval for single model: {args.model_filepath}.")
        submit_eval_for_model_filepath(args.model_filepath, args.username, validation_only, test_only)
        
    if args.model_dir:
        model_filepaths = identify_model_filepaths(args.model_dir)
        for model_filepath in model_filepaths:
            submit_eval_for_model_filepath(model_filepath, args.username, validation_only, test_only)
    
    logging.info("Finished.")
    

if __name__ == "__main__":
    main()
