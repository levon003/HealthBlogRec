# Process extracted from a notebook in this folder
# Given a sequence of date ranges, produces effect size estimates
# Implements the Bang-Robins estimator (from the Causal Inference book)

import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['font.family'] = "serif"
import json
import bson
from bson.codec_options import CodecOptions
from bson.raw_bson import RawBSONDocument
from bson import ObjectId
import gzip

import os
from tqdm import tqdm
import pickle
from glob import glob

from datetime import datetime
from dateutil.relativedelta import relativedelta
import dateutil
import pytz

import scipy
import scipy.stats

from pprint import pprint

from pathlib import Path
git_root_dir = '/panfs/roc/groups/1/lana/zentx005/repos/recsys-peer-match'

analysis_dir = os.path.join(git_root_dir, 'analysis')

import sys
caringbridge_core_path = "/home/lana/levon003/repos/caringbridge_core"
sys.path.append(caringbridge_core_path)
import cbcore.data.paths
assert os.path.exists(cbcore.data.paths.raw_data_filepath)
caringbridge_core_path = "/home/lana/levon003/repos/recsys-peer-match/src"
sys.path.append(caringbridge_core_path)
import cbrec.data


participant_data_dir = os.path.join(cbcore.data.paths.projects_data_dir, 'recsys-peer-match', 'participant')
#!wc -l {participant_data_dir}/*.ndjson

# load in recommendations from previous rounds
d = []
for batch_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    participant_data_filepath = os.path.join(participant_data_dir, f'participant_rec_data_b{batch_id}.ndjson')
    with open(participant_data_filepath, 'r') as infile:
        for line in infile:
            participant = json.loads(line)
            del participant['site_scores']
            participant['batch_id'] = batch_id
            d.append(participant)

batch_df = pd.DataFrame(d)

participant_recced_site_map = {}
for participant_id, group in batch_df.groupby('participant_id'):
    recced_site_ids = []
    for sse_site_list in group.sse_site_list:
        recced_site_ids.extend([site['site_id'] for site in sse_site_list])
    assert len(recced_site_ids) == len(set(recced_site_ids)), "Duplicate rec was given."
    recced_site_ids = list(set(recced_site_ids))
    participant_recced_site_map[participant_id] = recced_site_ids

recced_usps = [(row.participant_id, site['site_id']) for row in batch_df.itertuples() for site in row.sse_site_list]

assert len(set(recced_usps)) == len(recced_usps), "Duplicate rec given."

# create rec_df
rec_df = []
for row in batch_df.itertuples(index=False):
    for i, site in enumerate(row.sse_site_list):
        rec = row._asdict()
        del rec['sse_site_list']
        if 'journal_body' in site:
            # some of the data were written with different key names for cleaned_journal_{body,title}
            # this code normalizes the key names
            site = dict(site)
            site['cleaned_journal_body'] = site['journal_body']
            del site['journal_body']
            site['cleaned_journal_title'] = site['journal_title']
            del site['journal_title']
        rec.update(site)
        rec['rank'] = i
        rec_df.append(rec)
rec_df = pd.DataFrame(rec_df)

# add alias for participant_id
rec_df['user_id'] = rec_df['participant_id']

# get participant data
participant_id_filepath = os.path.join(git_root_dir, 'data/email/participant_ids.tsv')
participant_df = pd.read_csv(participant_id_filepath, sep='\t', header=0)

participant_batch_count_map = batch_df.groupby('participant_id').batch_id.nunique().to_dict()
participant_df['n_total_recs'] = participant_df.user_id.map(lambda user_id: participant_batch_count_map[user_id] * 5 if user_id in participant_batch_count_map else 0)

participant_first_sse_map = batch_df.groupby('participant_id').sse_sent_timestamp.min()
participant_df['first_sse_timestamp'] = participant_df.user_id.map(lambda user_id: participant_first_sse_map[user_id] if user_id in participant_first_sse_map else -1)

participant_user_ids = set(participant_df[participant_df.n_total_recs > 0].user_id)

control_sites_df = pd.read_csv(os.path.join(analysis_dir, "controlSites.csv")) 
control_site_ids = set(control_sites_df.site_id.unique())

actual_sites_df = pd.read_csv(os.path.join(analysis_dir, "actualSites.csv")) 
actual_site_ids = set(actual_sites_df.site_id.unique())

# load the site metadata dataframe
# this is created in caringbridge_core from the new data
site_metadata_working_dir = "/home/lana/shared/caringbridge/data/derived/site_metadata"
s = datetime.now()
site_metadata_filepath = os.path.join(site_metadata_working_dir, "site_metadata.feather")
site_info_df = pd.read_feather(site_metadata_filepath)
assert np.sum(site_info_df.site_id.value_counts() > 1) == 0, "Site ids are not globally unique."

# read the profile data
profile_metadata_dir = '/home/lana/shared/caringbridge/data/derived/profile'
s = datetime.now()
profile_df = pd.read_feather(os.path.join(profile_metadata_dir, 'profile.feather'))

# load the journal metadata
s = datetime.now()
journal_metadata_dir = "/home/lana/shared/caringbridge/data/derived/journal_metadata"
journal_metadata_filepath = os.path.join(journal_metadata_dir, "journal_metadata.feather")
journal_df = pd.read_feather(journal_metadata_filepath)

journal_df['usp'] = [(user_id, site_id) for user_id, site_id in zip(journal_df.user_id, journal_df.site_id)]

# read interactions dataframe
s = datetime.now()
model_data_dir = '/home/lana/shared/caringbridge/data/projects/recsys-peer-match/model_data'
ints_df = pd.read_feather(os.path.join(model_data_dir, 'ints_df.feather'))

ints_df['usp'] = [(user_id, site_id) for user_id, site_id in zip(ints_df.user_id, ints_df.site_id)]

# load the site profile diff
# rows should be >= 37M+
s = datetime.now()
site_profile_diff_filepath = os.path.join(cbcore.data.paths.projects_data_dir, 'caringbridge_core', 'site_profile_diff', 'site_profile_diff.tsv')
site_profile_diff_df = pd.read_csv(site_profile_diff_filepath, sep='\t', header=0)
print(f"Read {len(site_profile_diff_df)} rows in {datetime.now() - s}.")
site_profile_diff_df['usp'] = [(row.user_id, row.site_id) for row in tqdm(site_profile_diff_df.itertuples(), total=len(site_profile_diff_df), desc="Creating USPs")]

# also need to load the participant and non-participant site profile data

nonparticipant_data_dir = os.path.join(cbcore.data.paths.projects_data_dir, 'recsys-peer-match', 'nonparticipant')
with open(os.path.join(nonparticipant_data_dir, 'site_profile.pkl'), 'rb') as infile:
    nonp_site_profiles = pickle.load(infile)
print(len(nonp_site_profiles))

with open(os.path.join(participant_data_dir, 'site_profile.pkl'), 'rb') as infile:
    p_site_profiles = pickle.load(infile)
print(len(p_site_profiles))

site_profiles = nonp_site_profiles + p_site_profiles

# create a dataframe from the site profile entires
ds = []
for sp in site_profiles:
    user_id = int(sp['userId'])
    site_id = int(sp['siteId']) if 'siteId' in sp else -1
    # not capturing: nl
    d = {
        'user_id': user_id,
        'site_id': site_id,
        'is_creator': sp['isCreator'] if 'isCreator' in sp else None,
        'is_primary': sp['isPrimary'] if 'isPrimary' in sp else None,
        'role': sp['role'],
        'is_profile_deleted': sp['isProfileDeleted'] if 'isProfileDeleted' in sp else None,
        'is_site_deleted': sp['isSiteDeleted'] if 'isSiteDeleted' in sp else None,
        'is_stub': sp['isStub'] if 'isStub' in sp else None,
        'created_at': sp['createdAt'].timestamp() * 1000 if 'createdAt' in sp else 0,
        'updated_at': sp['updatedAt'].timestamp() * 1000 if 'updatedAt' in sp else 0,
        'n': dict(sp['n']) if 'n' in sp and sp['n'] is not None else {},
    }
    ds.append(d)

ssite_profile_df = pd.DataFrame(ds)
ssite_profile_df['is_recced'] = ssite_profile_df.site_id.isin(actual_site_ids)
ssite_profile_df['is_control'] = ssite_profile_df.site_id.isin(control_site_ids)
ssite_profile_df['usp'] = [(row.user_id, row.site_id) for row in ssite_profile_df.itertuples()]

ssite_profile_df.is_creator.value_counts(dropna=False)
ssite_profile_df.is_primary.value_counts(dropna=False)
ssite_profile_df['is_self_author'] = (ssite_profile_df.is_creator == 1)|(ssite_profile_df.is_primary == 1)|(ssite_profile_df.role == 'Organizer')

sjournal_df = journal_df[journal_df.user_id.isin(set(ssite_profile_df.user_id))]
journal_usp_set = set([(row.user_id, row.site_id) for row in sjournal_df.itertuples()])

# create the first_visit_df for others' sites only
first_visit_df = ssite_profile_df[~ssite_profile_df.is_self_author]
author_usp_set = set(ssite_profile_df[ssite_profile_df.is_self_author].usp) | set(journal_df.usp)
author_user_id_set = set(ssite_profile_df[ssite_profile_df.is_self_author].user_id) | set(journal_df.user_id)

# author-to-author site visits
# excludes all non-authors
# excludes all self-visits
site_visits = site_profile_diff_df[(site_profile_diff_df.key == 'updatedAt')&(site_profile_diff_df.user_id.isin(author_user_id_set)&(~site_profile_diff_df.usp.isin(author_usp_set)))]

user_site_interactions = {
    (row.user_id, row.site_id): [row.created_at,] for row in first_visit_df.itertuples()
}

TOLERANCE = 1000 * 60 * 60 * 7  # 7 hours, chosen so that if there's a bug with UTC (5 hours) and DST (1 hour) we still have an hour to treat them as essentially the same time

n_missing_site_profiles = 0
n_potential_missed_visits = 0
n_empty_curr_values = 0
for row in tqdm(site_visits.itertuples(), total=len(site_visits)):
    usp = (row.user_id, row.site_id)
    if usp not in user_site_interactions:
        # these are author interactions, but the author in question is not "eligible" i.e. not in the participant group or the pseudo-control group
        # the assertion below works as expected, although it requires running cells out of order
        # assert row.user_id not in target_user_ids
        n_missing_site_profiles += 1
        user_site_interactions[usp] = [float(row.old_value) * 1000,]
    visit_list = user_site_interactions[usp]
    last_visit = float(row.old_value) * 1000
    curr_visit = float(row.new_value) * 1000
    assert curr_visit > 0
    if last_visit == 0:
        n_empty_curr_values += 1
    elif last_visit < visit_list[-1] - TOLERANCE:
        logging.warning("updatedAt's old value was before the creation date of the site_profile or before the value from the previous snapshot.")
        break
    elif last_visit > visit_list[-1] + 5000:
        n_potential_missed_visits += 1
        visit_list.append(last_visit)
    assert curr_visit > last_visit
    visit_list.append(curr_visit)
    
visits_df = pd.DataFrame([{'usp': usp, 'visit_timestamp': visit_timestamp} for usp, visit_list in user_site_interactions.items() for visit_timestamp in visit_list])
visits_df['user_id'] = visits_df.usp.map(lambda usp: usp[0])
visits_df['site_id'] = visits_df.usp.map(lambda usp: usp[1])

visits_df['visit_date'] = visits_df.visit_timestamp.map(lambda ts: int(datetime.utcfromtimestamp(int(ts / 1000)).strftime('%Y%m%d')))

central_time = pytz.timezone('US/Central')
banner_live_time = datetime.fromisoformat('2021-08-02 12:11:00').astimezone(central_time)
banner_end_time = datetime.fromisoformat('2021-08-23 11:59:59').astimezone(central_time)

first_sse_timestamp = batch_df.sse_sent_timestamp.min()
first_sse_time = datetime.utcfromtimestamp(first_sse_timestamp / 1000)

last_sse_timestamp = batch_df.sse_sent_timestamp.max()
last_sse_time = datetime.utcfromtimestamp(last_sse_timestamp / 1000)

# load the rec_df with associated click data
participant_data_dir = '/home/lana/shared/caringbridge/data/projects/recsys-peer-match/participant'
click_rec_df = pd.read_feather(os.path.join(participant_data_dir, 'click_rec_df.feather'))

click_rec_df = click_rec_df[["participant_id", "site_id", "batch_id", "first_click_timestamp", "was_clicked"]]
click_rec_df['was_clicked'] = click_rec_df['was_clicked'].astype(int)

clicked_timestamps_df = click_rec_df[click_rec_df.was_clicked == 1].groupby('batch_id').first_click_timestamp.unique()

# group by site_id, was_clicked and first_click_timestamp = max(min(first_click_timestamp where was_clicked == 1), min(first_click_timestamp))
click_rec_sites_df = click_rec_df.groupby('site_id').apply(lambda x: pd.Series({'batch_id': min(x.batch_id), 'first_click_timestamp': max([x.first_click_timestamp.min(), x[x.was_clicked == 1].first_click_timestamp.min()]), 'was_clicked': x.was_clicked.max()}))

import random

random.seed(1)
#click_rec_df[~click_rec_df.was_clicked].first_click_timestamp = random.choice(clicked_timestamps_df[click_rec_df.batch_id])
click_rec_sites_df.first_click_timestamp = click_rec_sites_df[['batch_id','first_click_timestamp']].apply(lambda x: x.first_click_timestamp if x.first_click_timestamp != -1000 else random.choice(clicked_timestamps_df[x.batch_id]), axis = 1)
click_rec_sites_df.sort_values(by=['batch_id'])

click_control_sites_df = control_sites_df.groupby('site_id').apply(lambda x: pd.Series({'batch_id': min(x.first_batch), 'first_click_timestamp': random.choice(clicked_timestamps_df[min(x.first_batch)]), 'was_clicked': 0}))

random.seed(1)
#click_rec_df[~click_rec_df.was_clicked].first_click_timestamp = random.choice(clicked_timestamps_df[click_rec_df.batch_id])
click_rec_df.first_click_timestamp = click_rec_df[['batch_id','first_click_timestamp']].apply(lambda x: x.first_click_timestamp if x.first_click_timestamp != -1000 else random.choice(clicked_timestamps_df[x.batch_id]), axis = 1)

all_control_sites = pd.read_csv(os.path.join(analysis_dir, "allControlSites.csv")).astype(int)

random.seed(1)
all_control_sites['was_clicked'] = 0
all_control_sites['first_click_timestamp'] = all_control_sites[['batch_id']].apply(lambda x: random.choice(clicked_timestamps_df[x.batch_id]), axis = 1)

#click_rec_df = click_rec_df.set_index(['site_id','participant_id'])
all_control_sites = all_control_sites.set_index(['site_id','participant_id'])

target_site_ids = actual_site_ids | control_site_ids

sites_df = pd.concat([control_sites_df, actual_sites_df])

click_sites_df = pd.concat([click_control_sites_df, click_rec_sites_df[click_rec_sites_df.was_clicked==1]])

recced_usps = set([(row.participant_id, row.site_id) for row in rec_df.itertuples()])
recced_sites = set(rec_df.site_id)

import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

def compute_window_features(back_window, front_window, target_sites_df, exclude_participants=True):
    
    sjournal_df = target_sites_df.merge(journal_df[['site_id','published_at','user_id','journal_oid']], how='left', on='site_id')

    sjournal_df_pre = sjournal_df[(sjournal_df.first_click_timestamp - sjournal_df.published_at >= 0)&(sjournal_df.first_click_timestamp - sjournal_df.published_at <= back_window)]
    sjournal_df_post = sjournal_df[(sjournal_df.published_at - sjournal_df.first_click_timestamp >= 0)&(sjournal_df.published_at - sjournal_df.first_click_timestamp <= front_window)]

    n_updates_pre = sjournal_df_pre.groupby('site_id').journal_oid.nunique().rename("n_updates_pre")
    n_updates_post = sjournal_df_post.groupby('site_id').journal_oid.nunique().rename("n_updates_post")
    n_authors_pre = sjournal_df_pre.groupby('site_id').user_id.nunique().rename("n_authors_pre")
    n_authors_post = sjournal_df_post.groupby('site_id').user_id.nunique().rename("n_authors_post")
    
    #n_authors_total = journal_df[(journal_df.published_at <= end_timestamp)].groupby('site_id').user_id.nunique().rename("n_authors_total" + postfix) #Doesn't make sense for this analysis
    
    time_since_first_journal_update = target_sites_df.apply(lambda x: (x.first_click_timestamp - journal_df[(journal_df.site_id == x.name)].created_at.min()) / 1000 / 60 / 60 / 24, axis = 1).rename("time_since_first_journal_update")
    
    sints_df = target_sites_df.merge(ints_df[['site_id','usp','created_at','interaction_oid','user_id']], how='left', on='site_id')
    sints_df_pre = sints_df[(sints_df.first_click_timestamp - sints_df.created_at >= 0)&(sints_df.first_click_timestamp - sints_df.created_at <= back_window)]
    sints_df_post = sints_df[(sints_df.created_at - sints_df.first_click_timestamp >= 0)&(sints_df.created_at - sints_df.first_click_timestamp  <= front_window)]

    if exclude_participants:
        sints_df_pre = sints_df_pre[~sints_df_pre.usp.isin(recced_usps)]
        sints_df_post = sints_df_post[~sints_df_post.usp.isin(recced_usps)]
    is_self_interaction_pre = sints_df_pre.usp.isin(author_usp_set)
    is_self_interaction_post = sints_df_post.usp.isin(author_usp_set)

    interactionswith_pre = sints_df_pre[~is_self_interaction_pre].groupby(['site_id','usp']).interaction_oid.nunique()
    interactionswith_post = sints_df_post[~is_self_interaction_post].groupby(['site_id','usp']).interaction_oid.nunique()
    n_interactions_pre = interactionswith_pre.groupby('site_id').sum().rename("n_interactions_pre")
    n_interactions_post = interactionswith_post.groupby('site_id').sum().rename("n_interactions_post")
    n_users_interactedwith_pre = interactionswith_pre.groupby('site_id').count().rename("n_users_interactedwith_pre")
    n_users_interactedwith_post = interactionswith_post.groupby('site_id').count().rename("n_users_interactedwith_post")
    
    
    target_usps_pre = sints_df_pre[['user_id','site_id','interaction_oid']].merge(journal_df[['user_id','site_id']].drop_duplicates().rename(columns={'site_id': 'source_site_id'}), how='left', on='user_id')
    target_usps_post = sints_df_post[['user_id','site_id','interaction_oid']].merge(journal_df[['user_id','site_id']].drop_duplicates().rename(columns={'site_id': 'source_site_id'}), how='left', on='user_id')

    n_sitewide_interactionswith_pre = target_usps_pre[target_usps_pre.site_id != target_usps_pre.source_site_id]\
        .groupby(['source_site_id', 'user_id', 'site_id']).interaction_oid.nunique()
    n_sitewide_interactionswith_post = target_usps_post[target_usps_post.site_id != target_usps_post.source_site_id]\
        .groupby(['source_site_id', 'user_id', 'site_id']).interaction_oid.nunique()
    n_sitewide_interactionswith_self_pre = target_usps_pre[target_usps_pre.site_id == target_usps_pre.source_site_id]\
        .groupby(['source_site_id', 'user_id', 'site_id']).interaction_oid.nunique()
    n_sitewide_interactionswith_self_post = target_usps_post[target_usps_post.site_id == target_usps_post.source_site_id]\
        .groupby(['source_site_id', 'user_id', 'site_id']).interaction_oid.nunique()

    n_sitewide_interactions_pre = n_sitewide_interactionswith_pre.groupby('source_site_id').sum().rename("n_sitewide_interactions_pre")
    n_sitewide_interactions_post = n_sitewide_interactionswith_post.groupby('source_site_id').sum().rename("n_sitewide_interactions_post")
    n_sitewide_sites_intereactedwith_pre = n_sitewide_interactionswith_pre.groupby('source_site_id').count().rename("n_sitewide_sites_intereactedwith_pre")
    n_sitewide_sites_intereactedwith_post = n_sitewide_interactionswith_post.groupby('source_site_id').count().rename("n_sitewide_sites_intereactedwith_post")
    n_sitewide_self_interactions_pre = n_sitewide_interactionswith_self_pre.groupby('source_site_id').sum().rename("n_sitewide_self_interactions_pre")
    n_sitewide_self_interactions_post = n_sitewide_interactionswith_self_post.groupby('source_site_id').sum().rename("n_sitewide_self_interactions_post")
    
    
    sfirst_vist_df = target_sites_df.merge(first_visit_df[['site_id','user_id', 'usp', 'created_at']], how='left', on='site_id')
    sfirst_vist_df_pre = sfirst_vist_df[(sfirst_vist_df.first_click_timestamp - sfirst_vist_df.created_at >= 0)&(sfirst_vist_df.first_click_timestamp - sfirst_vist_df.created_at <= back_window)]
    sfirst_vist_df_post = sfirst_vist_df[(sfirst_vist_df.created_at - sfirst_vist_df.first_click_timestamp >= 0)&(sfirst_vist_df.created_at - sfirst_vist_df.first_click_timestamp <= front_window)]

    if exclude_participants:
        sfirst_vist_df_pre = sfirst_vist_df_pre[~sfirst_vist_df_pre.usp.isin(recced_usps)]
        sfirst_vist_df_post = sfirst_vist_df_post[~sfirst_vist_df_post.usp.isin(recced_usps)]

    n_first_visits_pre = sfirst_vist_df_pre.groupby('site_id').created_at.count().rename("n_first_visits_pre")
    n_first_visits_post = sfirst_vist_df_post.groupby('site_id').created_at.count().rename("n_first_visits_post")
    
    svisits_df = target_sites_df.merge(visits_df, how='left', on='site_id')
    svisits_df_pre = svisits_df[(svisits_df.first_click_timestamp - svisits_df.visit_timestamp >= 0)&(svisits_df.first_click_timestamp - svisits_df.visit_timestamp <= back_window)]
    svisits_df_post = svisits_df[(svisits_df.visit_timestamp - svisits_df.first_click_timestamp >= 0)&(svisits_df.visit_timestamp - svisits_df.first_click_timestamp <= front_window)]

    if exclude_participants:
        svisits_df_pre = svisits_df_pre[~svisits_df_pre.usp.isin(recced_usps)]
        svisits_df_post = svisits_df_post[~svisits_df_post.usp.isin(recced_usps)]


    n_days_visited_pre = svisits_df_pre.groupby('site_id').visit_date.nunique().rename("n_days_visited_pre")
    n_days_visited_post = svisits_df_post.groupby('site_id').visit_date.nunique().rename("n_days_visited_post")
    n_repeat_visits_pre = svisits_df_pre.groupby(['user_id', 'site_id']).visit_timestamp.count() - 1
    n_repeat_visits_post = svisits_df_post.groupby(['user_id', 'site_id']).visit_timestamp.count() - 1
    n_users_repeat_visited_pre = n_repeat_visits_pre[n_repeat_visits_pre > 0].groupby('site_id').count().rename("n_users_repeat_visited_pre")
    n_users_repeat_visited_post = n_repeat_visits_post[n_repeat_visits_post > 0].groupby('site_id').count().rename("n_users_repeat_visited_post")
    
    target_sites_df = target_sites_df.join([time_since_first_journal_update,
                  n_updates_pre,
                  n_updates_post,
                  n_authors_pre,
                  n_authors_post,
                  n_interactions_pre,
                  n_interactions_post,
                  n_users_interactedwith_pre,
                  n_users_interactedwith_post,
                  n_sitewide_interactions_pre,
                  n_sitewide_interactions_post,
                  n_sitewide_sites_intereactedwith_pre,
                  n_sitewide_sites_intereactedwith_post,
                  n_sitewide_self_interactions_pre,
                  n_sitewide_self_interactions_post,
                  n_first_visits_pre,
                  n_first_visits_post,
                  n_days_visited_pre,
                  n_days_visited_post,
                  n_users_repeat_visited_pre,
                  n_users_repeat_visited_post
    ])
    
    target_sites_df = target_sites_df.fillna(value=0)

    return target_sites_df
    
figures_dir = os.path.join(git_root_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

def generate_study_dataframes():
    one_day = 1000 * 60 * 60 * 24
    for time_window_days in tqdm(np.arange(7, 91 + 1, 7), desc='Weekly frame data'):#91
#         if time_window_days > 35:
#             continue
        back_window_days = 35 # min(time_window_days, 35)
        front_window_days = time_window_days

        # recced clicked vs non-clicked
        clicked_site_df = click_rec_sites_df[['first_click_timestamp', 'was_clicked']] 
        recced_df = compute_window_features(back_window_days * one_day, front_window_days * one_day, clicked_site_df)

        # recced clicked vs pseudo-control 
        recced_site_df = click_sites_df[['first_click_timestamp', 'was_clicked']]
        clicked_df = compute_window_features(back_window_days * one_day, front_window_days * one_day, recced_site_df)
            
        metadata = {
            'back_window_days': back_window_days,
            'front_window_days': front_window_days,
        }
        yield recced_df, clicked_df, metadata

import traceback
from sklearn.preprocessing import StandardScaler

def logit_ip_f(df, use_I=False):
    """
    Create the f(y|X) part of IP weights using logistic regression
    
    Adapted from https://github.com/jrfiedler/causal_inference_python_code/blob/master/chapter12.ipynb
    
    Parameters
    ----------
    df : Pandas DataFrame
    
    Returns
    -------
    Numpy array of IP weights
    
    """
    formula = """
        was_clicked ~ 
        np.log(time_since_first_journal_update) +
        n_updates_pre + 
        n_authors_pre +
        n_interactions_pre +
        n_users_interactedwith_pre + 
        n_sitewide_interactions_pre +
        n_sitewide_self_interactions_pre +
        n_sitewide_sites_intereactedwith_pre +
        n_first_visits_pre +
        n_days_visited_pre +
        n_users_repeat_visited_pre
    """
    model = smf.logit(formula=formula, data=df)
    res = model.fit(disp=0)
    #print(res.summary())
    weights = np.zeros(len(df))
    weights[df.was_clicked == 1] = res.predict(df[df.was_clicked == 1])
    weights[df.was_clicked == 0] = (1 - res.predict(df[df.was_clicked == 0]))
    return weights

def produce_ci_estimates(df, outcome):
    block2 = df.copy()
    block2.was_clicked = 0
    block3 = df.copy()
    block3.was_clicked = 1
    
    formula = outcome + """
        ~ was_clicked +
        np.log(time_since_first_journal_update) +
        n_updates_pre + 
        n_authors_pre +
        n_interactions_pre +
        n_users_interactedwith_pre + 
        n_sitewide_interactions_pre +
        n_sitewide_self_interactions_pre +
        n_sitewide_sites_intereactedwith_pre +
        n_first_visits_pre +
        n_days_visited_pre +
        n_users_repeat_visited_pre
    """
    
    raw_effect = df.loc[df.was_clicked==1, outcome].mean() - df.loc[df.was_clicked==0, outcome].mean()
    
    poisson_effect = -1
    poisson_ci = [-1, -1]
    if False:
        try:
            md = smf.glm(formula=formula, data=df, family=statsmodels.genmod.families.family.Poisson())
            res = md.fit(cov_type='HC0')
            if not res.mle_retvals['converged']:
                raise ValueError("Poisson model failed to converge.")
            poisson_effect = res.params.was_clicked
            poisson_ci = list(res.conf_int().loc['was_clicked'])
        except:
            poisson_effect = -1
            poisson_ci = [-1, -1]
    
    # basic regression estimates
    # that "adjust for" confounders
    # plus standardization
    md = smf.ols(formula=formula, data=df)
    res = md.fit()
    modeled_observational_effect = res.params.was_clicked
    modeled_observational_ci = list(res.conf_int().loc['was_clicked'])
    block2 = df.copy()
    block2.was_clicked = 0
    block3 = df.copy()
    block3.was_clicked = 1
    block2_pred = res.predict(block2)
    block3_pred = res.predict(block3)
    standardized_effect = block3_pred.mean() - block2_pred.mean()
    
    # IP weighting and the Bang-Robins doubly robust (DR) estimator
    weights = logit_ip_f(df)
    weights = 1 / weights
    wls = smf.wls(formula=f'{outcome} ~ was_clicked', data=df, weights=weights)
    res = wls.fit(disp=0)
    ip_weighted_effect = res.params.was_clicked
    
    block1 = df.copy()
    block1['R'] = weights
    block1.loc[block1.was_clicked == 0, 'R'] *= -1
    md = smf.ols(formula=formula + "+ R", data=block1)
    res = md.fit()
    block2 = block1.copy()
    block2.was_clicked = 0
    block3 = block1.copy()
    block3.was_clicked = 1
    block2_pred = res.predict(block2)
    block3_pred = res.predict(block3)
    dr_effect = block3_pred.mean() - block2_pred.mean()
    
    return {
        'raw_diff': raw_effect,
        'poisson_diff': poisson_effect,
        'poisson_ci': poisson_ci,
        'modeled_observational_diff': modeled_observational_effect,
        'modeled_observational_ci': modeled_observational_ci,
        'standardized_diff': standardized_effect,
        'ip_weighted_diff': ip_weighted_effect,
        'dr_diff': dr_effect,
    }

def compute_diff(df, outcome, bootstrap_iters=1000):
    ests = produce_ci_estimates(df, outcome)
    diff = {
        'outcome': outcome,
        'diff_raw': ests['raw_diff'],
        'diff_ols': ests['modeled_observational_diff'],
        'diff_ols_lower': ests['modeled_observational_ci'][0],
        'diff_ols_upper': ests['modeled_observational_ci'][1],
        'diff_poisson': ests['poisson_diff'],
        'diff_poisson_lower': ests['poisson_ci'][0],
        'diff_poisson_upper': ests['poisson_ci'][1],
        'diff_dr': ests['dr_diff'],
    }

    # bootstrapping
    bs_diffs = []
    for i in tqdm(range(bootstrap_iters), desc=f'Bootstrapping {outcome}', disable=True):
        sdf = df.sample(frac=1, replace=True)
        # TODO move this try/catch block into bsdiff, so that e.g. the raw and the OLS samples can still be computed
        try:
            ests = produce_ci_estimates(sdf, outcome)
        except Exception as e:
            
            continue
        bsdiff = {
            'diff_raw': ests['raw_diff'],
            'diff_ols': ests['modeled_observational_diff'],
            'diff_poisson': ests['poisson_diff'],
            'diff_dr': ests['dr_diff'],
        }
        bs_diffs.append(bsdiff)
    bsdiff_df = pd.DataFrame(bs_diffs)
    diff['n_bootstraps'] = len(bsdiff_df)
    for diff_col in ['diff_raw', 'diff_ols', 'diff_poisson', 'diff_dr']:
        means = bsdiff_df[diff_col]
        lower = means.quantile(0.025)
        upper = means.quantile(0.975)
        diff[diff_col + "_lower"] = lower
        diff[diff_col + "_upper"] = upper
        diff[diff_col + "_bs_means"] = list(means)
    return diff

def compute_effects():
    outcomes = [
        'n_updates_post', 
        'n_first_visits_post', 
        'n_users_repeat_visited_post', 
        'n_users_interactedwith_post', 
        'n_interactions_post', 
        'n_days_visited_post',
        'n_sitewide_interactions_post',
        'n_sitewide_sites_intereactedwith_post',
        'n_sitewide_self_interactions_post'
    ]
    diffs = []
    for recced_df, clicked_df, metadata in generate_study_dataframes():
        for time_period, df in (('recced', recced_df), ('clicked', clicked_df)):
            for outcome in tqdm(outcomes, desc='Outcomes'):
                diff = compute_diff(df, outcome)
                diff['time_period'] = time_period
                diff.update(metadata)
                diffs.append(diff)
    diff_df = pd.DataFrame(diffs)
    return diff_df

def compute_effects_test():
    outcomes = [
        'n_updates_post'
    ]
    diffs = []
    for recced_df, clicked_df, metadata in generate_study_dataframes():
        for time_period, df in (('recced', recced_df), ('clicked', clicked_df)):
            for outcome in tqdm(outcomes, desc='Outcomes'):
                diff = compute_diff(df, outcome)
                diff['time_period'] = time_period
                diff.update(metadata)
                diffs.append(diff)
    diff_df = pd.DataFrame(diffs)
    return diff_df

diff_df = compute_effects()

diff_df.to_feather("diff_df_20220528.feather")

from textwrap import wrap

# outcomes = [
#     'n_updates_post', 
#     'n_first_visits_post', 
#     'n_users_repeat_visited_post', 
#     'n_users_interactedwith_post', 
#     'n_interactions_post', 
#     'n_days_visited_post',
#     'n_sitewide_interactions_post',
#     'n_sitewide_sites_intereactedwith_post',
#     'n_sitewide_self_interactions_post'
# ]

outcomes = [ 'n_updates_post',  'n_updates_post']
pretty_name_map = {
    'n_updates_post': "Journal updates",
    'n_first_visits_post': "Peer visits",
    'n_users_repeat_visited_post': "Repeat user visits",
    'n_users_interactedwith_post': "Peer initiations", 
    'n_interactions_post': "Peer interactions", 
    'n_days_visited_post': "# days visiting peers",
    'n_sitewide_interactions_post': "Site author interactions",
    'n_sitewide_sites_intereactedwith_post': "Site author initiations",
    'n_sitewide_self_interactions_post': "Site author self interactions"
}

fig, axes = plt.subplots(len(outcomes), 2, figsize=(10, 22))

for time_period, col in zip(['recced', 'clicked'], [0, 1]):
    for row, outcome in enumerate(outcomes):
        ax = axes[row, col]
        sdf = diff_df[(diff_df.outcome == outcome)&(diff_df.time_period==time_period)]

        ax.axhline(0.0, color='black', linestyle='--')
        ax.axvline(5, color='gray', linestyle='-', alpha=0.5)

        fill_alpha = 0.05
        ax.plot(sdf.front_window_days / 7, sdf.diff_raw / sdf.front_window_days * 7, marker='.', label='Raw', color='blue')
        ax.fill_between(sdf.front_window_days / 7, sdf.diff_raw_lower / sdf.front_window_days * 7, sdf.diff_raw_upper / sdf.front_window_days * 7, color='blue', alpha=fill_alpha)

        ax.plot(sdf.front_window_days / 7, sdf.diff_ols / sdf.front_window_days * 7, marker='.', label='OLS', color='orange')
        ax.fill_between(sdf.front_window_days / 7, sdf.diff_ols_lower / sdf.front_window_days * 7, sdf.diff_ols_upper / sdf.front_window_days * 7, color='orange', alpha=fill_alpha)

        ax.plot(sdf.front_window_days / 7, sdf.diff_dr / sdf.front_window_days * 7, marker='.', label='DR', color='green')
        ax.fill_between(sdf.front_window_days / 7, sdf.diff_dr_lower / sdf.front_window_days * 7, sdf.diff_dr_upper / sdf.front_window_days * 7, color='green', alpha=fill_alpha)

        ax.set_xlabel(f"Time since clicked (weeks)")
        ax.set_ylabel("Excess weekly actions")
        ax.set_title("\n".join(wrap(f"{pretty_name_map[outcome]} after click ({'Clicked vs Psuedo Control' if time_period == 'recced' else 'Clicked vs Non-Clicked'})", 30)))
        ax.legend()

fig.tight_layout()
image_shortfilename = f"recced_site_outcomes_all.pdf"
image_filename = os.path.join(figures_dir, image_shortfilename)
fig.savefig(image_filename, format='pdf', dpi=200, pad_inches=0, bbox_inches='tight') #, transparent=True)
plt.show()
sdf.head()