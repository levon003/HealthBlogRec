import pandas as pd
from tqdm import tqdm
import json


def stream_metadata_list(filepath):
    with open(filepath, 'r') as infile:
        for line in tqdm(infile, desc='Streaming metadata file'):
            md = json.loads(line)
            
            if md['is_test_period'] and md['is_initiation_eligible'] and md['has_target']:
                md['type'] = 'test'
            elif md['is_test_period'] and md['is_initiation_eligible'] and not md['has_target']:
                md['type'] = 'predict'
            elif not md['is_test_period'] and md['is_initiation_eligible'] and md['has_target']:
                md['type'] = 'train'
            elif not md['is_initiation_eligible']:
                md['type'] = 'ineligible'
            else:
                md['type'] = 'other'
            yield md


def get_metadata_list(filepath):
    md_list = list(stream_metadata_list(filepath))
    return md_list


def get_test_metadata(md_list):
    return [md for md in md_list if md['type'] == 'test']


def create_metadata_dataframe(md_list):
    df = pd.DataFrame(md_list, columns=[
            'type',  # special column, created by metadata loading logic (see get_metadata_list)
            'metadata_id', 
            'timestamp',
            'source_user_id',
            'target_site_id',
            'is_test_period',
            'n_source_sites',
            'n_target_users',
            'source_user_is_existing',
            'n_existing_users_on_target_site',
            'source_user_is_eligible',
            'target_site_has_eligible_user',
            'is_self_initiation',
            'is_initiation_eligible',  
            'has_target',
            # and the features that come with being initiation eligible...
            'n_eligible_users',
            'n_eligible_coauthors',
            'n_source_usps',
            'n_active_user_ids',
            'source_user_is_active',
            'n_active_target_users',
            'n_target_usps',
            'n_eligible_inactive_users',
            'n_existing_initiations_from_source_user_id',
            'n_candidate_user_ids',
            'n_candidate_usps',
            # test-only features
            'test_target_usp_adjustment',
            'source_user_initiated_in_train_period',
            'target_site_initiated_with_in_train_period',
        ]
    )
    return df
