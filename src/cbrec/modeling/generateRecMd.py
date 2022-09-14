#!/usr/bin/env python3
# Script to evaluate models
"""
Generates the test RecContext objects as pickle files in Global Scratch storage.

Global storage info: https://www.msi.umn.edu/content/scratch-storage
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


def generate_rec_contexts(config, allow_errors=False):
    logger = logging.getLogger("cbrec.modeling.generateRecMd.generate_rec_contexts")

    text_loader = cbrec.modeling.text_loader.TextLoader(config)
    text_loader.cache_all_journals()
    logger.info("TextLoader created and embeddings cached.")
    metadata_dir = "/scratch.global/levon003/rec_context_md/"
    os.makedirs(metadata_dir, exist_ok=True)
    test_md_list = [md for md in cbrec.utils.stream_metadata_list(config.metadata_filepath) if md['type'] == 'test' or md['type'] == 'predict']
    test_md_map = {md['metadata_id']: md for md in test_md_list}
    db = cbrec.featuredb.get_db_by_filepath(config.feature_db_filepath)
    count = 0
    n_generation_errors = 0
    for test_context in tqdm(cbrec.featuredb.stream_test_contexts(db, config), desc='Streaming test contexts', total=len(test_md_map)):
        test_context_md = test_md_map[test_context['metadata_id']]
        try:
            rc = cbrec.modeling.reccontext_builder.build_reccontext(config, text_loader, test_context_md, test_context)
            metadata_filepath = os.path.join(metadata_dir, f"reccontext{count}.pkl")
            with open(metadata_filepath, 'wb') as reccontext_file:   
                pickle.dump(rc, reccontext_file)
            count = count + 1
        except ValueError as ex:
            n_generation_errors += 1
            if not allow_errors:
                raise ex
    logger.info(f"Finished writing all rec context files. ({n_generation_errors} generation errors)")


def main():        
    config = cbrec.genconfig.Config()
    generate_rec_contexts(config)
    

if __name__ == "__main__":
    main()
