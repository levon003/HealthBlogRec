#!/usr/bin/env python3
# Script to flatten BSON into JSON, extracting only particular fields
# gonna be honest, don't remember what we used these scripts for

import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

import os
import re
import sys
import pandas as pd
import numpy as np
import argparse

from tqdm import tqdm
import gzip
import json
import bson
from bson.codec_options import CodecOptions
from bson.raw_bson import RawBSONDocument

from datetime import datetime
import pytz

try:
    import cbcore.data.paths
except:
    logging.warning("Manually adding cbcore directory to path.")
    caringbridge_core_path = "/home/lana/levon003/repos/caringbridge_core"
    sys.path.append(caringbridge_core_path)
    import cbcore.data.paths
    
import cbcore.bson.decode
import cbcore.data.dates as dates
from cbcore.data.utils import extract_long
from cbcore.script.computeCollectionCounts import iterate_collection


def process_comment_json_gz(output_filepath, valid_user_ids, valid_site_ids):
    input_filepath = os.path.join(cbcore.data.paths.raw_data_filepath, 'comment_scrubbed.json.gz')
    assert os.path.exists(input_filepath)
    logging.info(f"Will process from '{input_filepath}'.")
    
    processed_count = 0
    with gzip.open(input_filepath, 'rt', encoding='utf-8') as infile, open(output_filepath, 'w') as outfile:
        for i, line in tqdm(enumerate(infile), total=46619495):
            comment = json.loads(line)
            #comment_oid = comment['_id']['$oid']
            #parent_type = comment['parentType']  # either 'journal' or 'comment'
            #parent_oid = comment['parentId']
            #journal_oid = comment['ancestorId']  # ancestorType is never guestbook; you currently can't comment on guestbooks
            site_id = extract_long(comment['siteId'])
            user_id = extract_long(comment['userId'])
            if user_id in valid_user_ids or site_id in valid_site_ids:
                outfile.write(json.dumps(comment) + "\n")
            processed_count += 1
    logging.info(f"Processed {processed_count} lines.")
    

def process_guestbook_json_gz(output_filepath, valid_user_ids, valid_site_ids):
    input_filepath = os.path.join(cbcore.data.paths.raw_data_filepath, 'guestbook_scrubbed.json.gz')
    assert os.path.exists(input_filepath)
    logging.info(f"Will process from '{input_filepath}'.")
    
    processed_count = 0
    with gzip.open(input_filepath, 'rb') as infile, open(output_filepath, 'w') as outfile:
        for i, line in tqdm(enumerate(infile), total=82623921):
            if i < 4002:
                continue
            try:
                gb = json.loads(line.decode('utf-8'))
            except:
                logging.warning(f"Failed decoding line {i}: '{line}'")
                n_format_errors += 1
                continue
            if '_id' not in gb or '$oid' not in gb['_id']:
                #logging.warning(f"No id on line {i}: '{gb}'")
                continue
            gb_oid = gb['_id']['$oid']
            site_id = extract_long(gb['siteId'])
            user_id = extract_long(gb['userId'])
            #created_at = dates.get_date_from_json_value(gb['createdAt']) if 'createdAt' in gb else 0
            #updated_at = dates.get_date_from_json_value(gb['updatedAt']) if 'updatedAt' in gb else 0
            if user_id in valid_user_ids or site_id in valid_site_ids:
                outfile.write(json.dumps(gb) + "\n")
            processed_count += 1
    logging.info(f"Processed {processed_count} lines.")
            

def process_site_profile_json_gz(output_filepath, valid_user_ids, valid_site_ids):
    input_filepath = os.path.join(cbcore.data.paths.raw_data_filepath, 'site_profile.bson.gz')
    assert os.path.exists(input_filepath)
    logging.info(f"Will process from '{input_filepath}'.")
    
    SITE_PROFILE_DICT_KEYS = set(['c', 'n', 'my'])
    SITE_PROFILE_DATETIME_KEYS = set(['refAt', 'note', 'nlu'])
    VALID_KEYS = set(['_id', 'createdAt', 'guid', 'updatedAt', 'userId', 'role', 'signature', 'siteId', 'guid', 'nl', 'isCreator', 'isPrimary', 'isProfileDeleted', 'isSiteDeleted', 'isStub']) | SITE_PROFILE_DICT_KEYS | SITE_PROFILE_DATETIME_KEYS
    
    processed_count = 0
    with open(output_filepath, 'w') as outfile: 
        for doc in tqdm(iterate_collection(input_filepath), desc='Processing site profiles'):
            user_id = int(doc['userId']) if 'userId' in doc else -1
            site_id = int(doc['siteId']) if 'siteId' in doc else -1
            if user_id in valid_user_ids or site_id in valid_site_ids:
                d = {key: value for key, value in doc.items() if key in VALID_KEYS}
                d['_id'] = str(d['_id'])
                d['createdAt'] = int(d['createdAt'].timestamp() * 1000) if 'createdAt' in d else -1
                d['updatedAt'] = int(d['updatedAt'].timestamp() * 1000) if 'updatedAt' in d else -1
                if 'nl' in d:
                    d['nl'] = [dict(nl) for nl in d['nl']]
                keys = set(d.keys())
                
                datetime_keys = SITE_PROFILE_DATETIME_KEYS & keys
                for key in datetime_keys:
                    d[key] = int(d[key].timestamp() * 1000)
                
                dict_keys = SITE_PROFILE_DICT_KEYS & keys
                for key in dict_keys:
                    d[key] = dict(d[key])
                try:
                    outfile.write(json.dumps(d) + "\n")
                except:
                    print(d)
                    raise ValueError()
            processed_count += 1
    logging.info(f"Processed {processed_count} lines.")


def process_reactions_json_gz(output_filepath, valid_user_ids, valid_site_ids):
    input_filepath = os.path.join(cbcore.data.paths.raw_data_filepath, 'reaction.bson.gz')
    assert os.path.exists(input_filepath)
    logging.info(f"Will process from '{input_filepath}'.")
    
    options = CodecOptions(document_class=RawBSONDocument, tz_aware=True, tzinfo=pytz.UTC)
    processed_count = 0
    with gzip.open(input_filepath, 'rb') as infile, open(output_filepath, 'w') as outfile:
        doc_iter = bson.decode_file_iter(infile, codec_options=options)
        for doc in tqdm(doc_iter, total=1009974):
            reaction = cbcore.bson.decode.inflate_raw_bson(doc, options)
            
            #reaction_oid = str(reaction['_id'])
            user_id = int(reaction['reactorId'])
            site_id = int(reaction['siteId'])
            if user_id in valid_user_ids or site_id in valid_site_ids:
                d = dict(reaction)
                d['_id'] = str(d['_id'])
                d['createdAt'] = int(d['createdAt'].timestamp() * 1000)
                del d['reactorImagePath']
                outfile.write(json.dumps(d) + "\n")
            processed_count += 1
    logging.info(f"Processed {processed_count} lines.")


def load_valid_user_ids(activity_dir):
    valid_user_ids, valid_site_ids = set(), set()
    for fname, id_set in zip(('valid_user_ids.txt', 'valid_site_ids.txt'), (valid_user_ids, valid_site_ids)):
        with open(os.path.join(activity_dir, fname), 'r') as infile:
            for line in infile:
                if line.strip() != "":
                    id_set.add(int(line.strip()))
    return valid_user_ids, valid_site_ids
    
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection-name', dest='collection_name', required=True)
    args = parser.parse_args()

    activity_dir = os.path.join(cbcore.data.paths.projects_data_filepath, 'recsys-peer-match', 'activity')
    
    output_filepath = os.path.join(activity_dir, f'{args.collection_name}.ndjson')
    logging.info(f"Will write output to '{output_filepath}'.")
    
    valid_user_ids, valid_site_ids = load_valid_user_ids(activity_dir)
    valid_site_ids
    
    if args.collection_name == 'reaction':
        process_reactions_json_gz(output_filepath, valid_user_ids, valid_site_ids)
    elif args.collection_name == 'site_profile':
        process_site_profile_json_gz(output_filepath, valid_user_ids, valid_site_ids)
    elif args.collection_name == 'guestbook':
        process_guestbook_json_gz(output_filepath, valid_user_ids, valid_site_ids)
    elif args.collection_name == 'comment':
        process_comment_json_gz(output_filepath, valid_user_ids, valid_site_ids)
    else:
        raise ValueError(f"Not yet implemented {args.collection_name}")
    logging.info(f"Finished.")
    
    
if __name__ == '__main__':
    main()
