import os
import sys
import pandas as pd
import numpy as np

from collections import defaultdict
from tqdm import tqdm
import pickle
import json

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytz

# import the cbcore library
caringbridge_core_path = "/home/lana/levon003/repos/caringbridge_core"
sys.path.append(caringbridge_core_path)
import cbcore.data.paths as paths
import cbcore.data.dates as dates
import cbcore.data.utils as utils

working_dir = "/home/lana/shared/caringbridge/data/projects/recsys-peer-match/prerec_evidence"

user_id_blocklist = [
    0,  # Test user
    1,  # Weird user
    15159562,  # Test user account run by CaringBridge Customer Experience team
    46,  # Seems to be run at least in part by CaringBridge team for testing
    13896060,  # Seems like another customer care rep
    594,  # Seems like a customer care rep, but also seems like it may include some legitimate sites? (see e.g. site 559205)
    7393709, #Junk and test posts
    25036137, #Repeated test text
    8192483, #Mostly test posts, but one genuine post about patient
    17956362, #Test posts
    16648084, #Test posts (and some good poetry)
    31761432, # Doctor's ad
    32764680, # Payday lending ad
    30457719, # 3D visualization company ad
    32538830, # Car supplies ad
    32757690, # Fashion ad
    32757739, # Clothing brand ad
    1043681, # Leasing furniture ad
    28132146, # Farm company ad
    31477721, # Lenders ad
    31879875, # Payday lender ad
    31799168, # Credit company ad
    32428328, # SEO ad
    31684805, # Various ads
    30165532, # Various ads about black magic
    31833912, # Job hunting spam
    32753111, # Arabic text (possibly spam)
    32732132 # Turkish text (spam)
]

EARLY_SITE_COUNT_THRESHOLD_MS = 1000 * 60 * 60 * 24 * 30  # 30 days
INVALID_END_DATE_STR = '2021-08-01'


def get_data():
    # load the user site dataframe
    s = datetime.now()
    model_data_dir = '/home/lana/shared/caringbridge/data/projects/recsys-peer-match/model_data'
    user_site_df = pd.read_csv(os.path.join(model_data_dir, 'user_site_df.csv'))
    valid_user_ids = set(user_site_df.user_id)
    valid_site_ids = set(user_site_df.site_id)
    print(f"Read {len(user_site_df)} user_site_df rows ({len(valid_user_ids)} unique users, {len(valid_site_ids)} unique sites) in {datetime.now() - s}.")
    
    # load the journal dataframe
    s = datetime.now()
    journal_metadata_dir = "/home/lana/shared/caringbridge/data/derived/journal_metadata"
    journal_metadata_filepath = os.path.join(journal_metadata_dir, "journal_metadata.feather")
    journal_df = pd.read_feather(journal_metadata_filepath)
    print(f"Read {len(journal_df)} journal_df rows in {datetime.now() - s}.")

    return user_site_df, journal_df


def build_start_timestamp_dicts(journal_df, invalid_end_date_str=INVALID_END_DATE_STR):
    # identify site first update timestamps
    s = datetime.now()
    site_start_dict = journal_df.groupby('site_id').created_at.min().to_dict()
    # ignore sites with weird start times
    invalid_start_date = datetime.fromisoformat('2005-01-01').replace(tzinfo=pytz.UTC)
    invalid_end_date = datetime.fromisoformat(invalid_end_date_str).replace(tzinfo=pytz.UTC)
    print(f"Keeping sites between {invalid_start_date.isoformat()} and {invalid_end_date.isoformat()}.")
    invalid_start_timestamp = invalid_start_date.timestamp() * 1000
    invalid_end_timestamp = invalid_end_date.timestamp() * 1000
    deleted_site_count = 0
    for site_id in list(site_start_dict.keys()):
        site_start_timestamp = site_start_dict[site_id]
        if site_start_timestamp <= invalid_start_timestamp or site_start_timestamp >= invalid_end_timestamp:
            deleted_site_count += 1
            del site_start_dict[site_id]
    print(f"Identified {len(site_start_dict)} site first update times in {datetime.now() - s} (removed {deleted_site_count} sites for invalid start times).")
    
    # identify user first update timestamps
    s = datetime.now()
    author_start_dict = journal_df.groupby('user_id').created_at.min().to_dict()
    # ignore sites with weird start times
    deleted_user_count = 0
    for user_id in list(author_start_dict.keys()):
        user_start_timestamp = author_start_dict[user_id]
        if user_start_timestamp <= invalid_start_timestamp or user_start_timestamp >= invalid_end_timestamp:
            deleted_user_count += 1
            del author_start_dict[user_id]
    print(f"Identified {len(author_start_dict)} user first update times in {datetime.now() - s} (removed {deleted_user_count} users for invalid start times).")
    
    return site_start_dict, author_start_dict


def compute_counts():
    user_site_df, journal_df = get_data()
    
    # filter the journal_df
    valid_user_ids = set(user_site_df.user_id)
    journal_df = journal_df[journal_df.user_id.isin(valid_user_ids)]
    invalid_start_date = datetime.fromisoformat('2005-01-01').replace(tzinfo=pytz.UTC)
    invalid_end_date = datetime.fromisoformat(INVALID_END_DATE_STR).replace(tzinfo=pytz.UTC)
    print(f"Keeping journals between {invalid_start_date.isoformat()} and {invalid_end_date.isoformat()}.")
    invalid_start_timestamp = invalid_start_date.timestamp() * 1000
    invalid_end_timestamp = invalid_end_date.timestamp() * 1000
    journal_df = journal_df[(journal_df.created_at>=invalid_start_timestamp)&(journal_df.created_at<=invalid_end_timestamp)]
    print(f"New journal_df length = {len(journal_df)}")

    site_start_dict, author_start_dict = build_start_timestamp_dicts(journal_df)
    
    s = datetime.now()
    site_author_map = user_site_df.groupby('site_id').user_id.agg(lambda g: set(g)).to_dict()
    print(f"Built {len(site_author_map)} site_author_map (site_id -> set(user_id) dict) in {datetime.now() - s}.")
    
    count_interactions(site_start_dict, author_start_dict, site_author_map)
    
    
def save_counts(file_key, site_early_int_count, site_early_author_int_count, site_early_self_int_count):
    counts_dir = os.path.join(working_dir, 'early_site_int_counts')
    os.makedirs(counts_dir, exist_ok=True)
    
    for fname_root, count_dict in zip(['site_early_int_count', 'site_early_author_int_count', 'site_early_self_int_count'], [site_early_int_count, site_early_author_int_count, site_early_self_int_count]):
        output_filepath = os.path.join(counts_dir, f'{file_key}_{fname_root}.pkl')
        with open(output_filepath, 'wb') as outfile:
            pickle.dump(count_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {fname_root} to '{output_filepath}'.")
    
    
def load_counts(file_key, as_dataframe=True):
    counts_dir = os.path.join(working_dir, 'early_site_int_counts')
    assert os.path.exists(counts_dir)
    count_dicts = []
    for fname_root in ['site_early_int_count', 'site_early_author_int_count', 'site_early_self_int_count']:
        count_dict_filepath = os.path.join(counts_dir, f'{file_key}_{fname_root}.pkl')
        with open(count_dict_filepath, 'rb') as infile:
            count_dict = pickle.load(infile)
            count_dicts.append(count_dict)
    if not as_dataframe:
        return count_dicts
    else:
        nonzero_site_ids = set(count_dicts[0].keys()) | set(count_dicts[1].keys()) | set(count_dicts[2].keys())
        columns = [f'n_early_int_{file_key}', f'n_early_author_int_{file_key}', f'n_early_self_int_{file_key}']
        int_count_df = pd.DataFrame(data=0, index=nonzero_site_ids, columns=columns, dtype=int)
        for site_id in int_count_df.index:
            for i, col in enumerate(columns):
                if site_id in count_dicts[i]:
                    int_count_df.at[site_id, col] = count_dicts[i][site_id]
        return int_count_df


def count_interactions(site_start_dict, author_start_dict, site_author_map):
    int_types = ['amp', 'comment', 'guestbook']
    site_early_int_count = {int_type: defaultdict(int) for int_type in int_types}
    site_early_author_int_count = {int_type: defaultdict(int) for int_type in int_types}
    site_early_self_int_count = {int_type: defaultdict(int) for int_type in int_types}  # note: self interactions are tracked seperately
    
    interactions_dir = '/home/lana/shared/caringbridge/data/derived/interactions'
    for filename in ['reaction.csv', 'amps.csv', 'comment.csv', 'guestbook.csv']:
        blocked_user_int_count = 0
        processed_count = 0
        input_filepath = os.path.join(interactions_dir, filename)
        with open(input_filepath, 'r') as infile:
            for line in tqdm(infile, desc=filename):
                processed_count += 1
                # columns: user_id, site_id, interaction_type, interaction_oid, parent_type, parent_id, ancestor_type, ancestor_id, created_at, updated_at
                tokens = line.strip().split(",")
                user_id = int(tokens[0])
                site_id = int(tokens[1])
                interaction_type = tokens[2]
                created_at = int(tokens[-2]) if tokens[-2] != "" else -1
                
                if site_id not in site_start_dict or site_id not in site_author_map:
                    #  not a site we care about
                    continue
                int_timestamp = created_at
                
                is_int_within_early_site_threshold = int_timestamp <= site_start_dict[site_id] + EARLY_SITE_COUNT_THRESHOLD_MS
                if not is_int_within_early_site_threshold:
                    # not early site interaction; we don't care about these
                    # note: particularly fraught metric for amps, since any amp on a journal within EARLY_SITE_COUNT_THRESHOLD_MS of the site start will be considered to be early activity
                    continue
                    
                int_type = interaction_type
                if int_type.startswith("amp"):
                    int_type = "amp"

                site_author_ids = site_author_map[site_id]
                if user_id in user_id_blocklist:
                    blocked_user_int_count += 1
                    continue
                is_self_interaction = user_id in site_author_ids
                if is_self_interaction:
                    site_early_self_int_count[int_type][site_id] += 1
                    continue
                
                is_user_author = user_id in author_start_dict and int_timestamp >= author_start_dict[user_id]
                site_early_int_count[int_type][site_id] += 1
                if is_user_author:
                    site_early_author_int_count[int_type][site_id] += 1
        print(f"{filename}: While computing counts from {processed_count} lines, identified {blocked_user_int_count} interactions from blocked users.")
    for int_type in int_types:
        save_counts(int_type, site_early_int_count[int_type], site_early_author_int_count[int_type], site_early_self_int_count[int_type])
    
    
def main():
    compute_counts()
    

if __name__ == '__main__':
    main()
