"""
Recovery script, for when things are bad with the database.
"""

import os
import sys
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import argparse

try:
    import cbrec
except:
    sys.path.append("/home/lana/levon003/repos/recsys-peer-match/src")

import cbrec.featuredb
import cbrec.genconfig
import cbrec.utils
import cbrec.logutils

def get_test_metadata_ids(config):
    metadata_ids = set()
    for md in cbrec.utils.stream_metadata_list(config.metadata_filepath):
        if md['type'] == 'test':
            metadata_ids.add(md['metadata_id'])
    return metadata_ids

def delete_test_contexts(dry_run):
    """
    Deletes all test contexts from the database.
    
    After running this, must also remove the offending entries from the metadata file:
        grep -v 'has_target": true.*is_test_period": true.*is_initiation_eligible":' metadata.ndjson_with_deleted_test_contexts > metadata.ndjson
    
    """
    logger = logging.getLogger('cbrec.recover.delete_test_contexts')
    config = cbrec.genconfig.Config()
    
    metadata_ids = get_test_metadata_ids(config)
    logger.info(f"Identified {len(metadata_ids)} test_contexts.")
    
    
    tosave_feature_id_counts = defaultdict(int)  # map of feature_id -> count
    delete_feature_id_counts = defaultdict(int)  # map of feature_id -> count
    seen_metadata_ids = set()
    db = cbrec.featuredb.get_feature_db(config)
    with db:
        cursor = db.execute("""
        SELECT 
            metadata_id, 
            source_usp_arr_id,
            candidate_usp_arr_id,
            target_inds_id,
            source_usp_mat_id,
            candidate_usp_mat_id,
            user_pair_mat_id
        FROM test_context
        """)
        if cursor is None:
            return ValueError("Null cursor.")
        row = cursor.fetchone()
        while row is not None:
            row_md = {key: row[key] for key in row.keys()}
            if row_md['metadata_id'] in metadata_ids:
                count_dict = delete_feature_id_counts
                seen_metadata_ids.add(row_md['metadata_id'])
            else:
                count_dict = tosave_feature_id_counts
            for feature_id_key in ['source_usp_arr_id', 'candidate_usp_arr_id', 'target_inds_id', 'source_usp_mat_id', 'candidate_usp_mat_id', 'user_pair_mat_id']:
                feature_id = row_md[feature_id_key]
                count_dict[feature_id] += 1
            row = cursor.fetchone()
    if len(seen_metadata_ids) != len(metadata_ids):
        logger.info(f"Note: {len(seen_metadata_ids)} of {len(metadata_ids)} metadata_ids actually found in test_context table.")
    
    feature_ids_to_delete = []
    for key in delete_feature_id_counts.keys():
        if key not in tosave_feature_id_counts:
            feature_ids_to_delete.append(key)
    logger.info(f"Will delete {len(feature_ids_to_delete)} / {len(delete_feature_id_counts)} feature arrays in addition to {len(metadata_ids)} test_contexts.")
    
    if dry_run:
        logger.info("Dry run; terminating.")
        return
    db = cbrec.featuredb.get_feature_db(config)
    with db:
        db.executemany("DELETE FROM test_context WHERE metadata_id = ?", [(v,) for v in seen_metadata_ids])
        logger.info("Deleted test_contexts.")
        db.executemany("DELETE FROM feature WHERE feature_id = ?", [(v,) for v in feature_ids_to_delete])
        logger.info("Deleted features.")
        db.commit()
    logger.info("Deletion complete.")
    

def main():
    cbrec.logutils.set_up_logging()
    logger = logging.getLogger('cbrec.recover.main')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', dest='dry_run', action='store_true', default=False)
    args = parser.parse_args()
    
    # Deletes all test contexts (NOT including predictions)
    #delete_test_contexts(args.dry_run)

    logger.info("Finished.")
    
    
if __name__ == '__main__':
    main()
