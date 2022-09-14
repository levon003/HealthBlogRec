"""

Prediction encompasses several tasks.

For now, the question is: can I get the baseline scores and ranks for a single user_id?

21 Aug 2021: 1629883002440
--timestamp 1630547604000 --checkpoint-name rdg_35916870_736346.pkl --user-ids-file /home/lana/shared/caringbridge/data/projects/recsys-peer-match/participant/eligible_participant_user_ids.txt

15 Sept 2021:
--timestamp 1631802968000 --checkpoint-name rdg_36028292_734622.pkl --user-ids-file /home/lana/shared/caringbridge/data/projects/recsys-peer-match/participant/eligible_participant_user_ids.txt

23 Sept 2021:
--timestamp infer --checkpoint-name rdg_36175077_849287.pkl --user-ids-file /home/lana/shared/caringbridge/data/projects/recsys-peer-match/participant/eligible_participant_user_ids.txt
inferred 1632406020496

29 Sept 2021:
--timestamp infer --checkpoint-name rdg_36261144_851080.pkl --user-ids-file /home/lana/shared/caringbridge/data/projects/recsys-peer-match/participant/eligible_participant_user_ids.txt
inferred 1633029219763

20 October 2021:
--timestamp infer --checkpoint-name rdg_36475519_856096.pkl --user-ids-file /home/lana/shared/caringbridge/data/projects/recsys-peer-match/participant/eligible_participant_user_ids.txt
inferred 

27 October 2021:
--timestamp infer --checkpoint-name rdg_36554748_857815.pkl --user-ids-file /home/lana/shared/caringbridge/data/projects/recsys-peer-match/participant/eligible_participant_user_ids.txt
inferred 1635362744684

2 November 2021:
--timestamp infer --checkpoint-name rdg_36618193_859181.pkl --user-ids-file /home/lana/shared/caringbridge/data/projects/recsys-peer-match/participant/eligible_participant_user_ids.txt
inferred 1635884125060

10 November 2021:
--timestamp infer --checkpoint-name rdg_36709922_861360.pkl --user-ids-file /home/lana/shared/caringbridge/data/projects/recsys-peer-match/participant/eligible_participant_user_ids.txt
inferred 1636591362069

16 November 2021:
--timestamp infer --checkpoint-name rdg_36788915_863172.pkl --user-ids-file /home/lana/shared/caringbridge/data/projects/recsys-peer-match/participant/eligible_participant_user_ids.txt
inferred 1637185880982

24 November 2021:
--timestamp infer --checkpoint-name rdg_36872012_866861.pkl --user-ids-file /home/lana/shared/caringbridge/data/projects/recsys-peer-match/participant/eligible_participant_user_ids.txt
inferred 1637790059133

"""
from . import logutils
from . import genconfig
from . import testutils
from . import triple_generation
from .text import journalid

import os
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime


def get_rec_data_generator(config, filename):
    #triple_generation.RecDataGenerator
    checkpoint_filepath = os.path.join(config.checkpoint_dir, filename)
    with open(checkpoint_filepath, 'rb') as infile:
        rdg = pickle.load(infile)
    #rdg.database_writer.stop_process()  # no easy way to stop the process from starting in the first place, so we just shut it down now
    return rdg


def score_summary():
    # Deprecated / unknown functionality
    model_ranks = {model_name: scorer.compute_ranks(y_score_site) for model_name, y_score_site in scorer.scores_dict.items()}
    logger.info("Score summary:")
    for i, site_id in enumerate(scorer.site_id_arr):
        line = f"Site id={site_id}\n"
        for model_name in scorer.scores_dict.keys():
            y_score_site = scorer.scores_dict[model_name]
            ranks = model_ranks[model_name]
            line += f"    {model_name}={y_score_site[i]:.2f} {ranks[i]}\n"
        logger.info(line)


def get_user_ids(args):
    logger = logging.getLogger('cbrec.prediction.get_user_ids')
    user_id_list = []
    if args.user_id_list.strip() != '':
        tokens = args.user_id_list.strip().split(",")
        user_ids = [int(user_id) for user_id in tokens]
        logger.info(f"Read {len(user_ids)} user ids from the command line.")
        user_id_list.extend(user_ids)
    if args.user_id_filepath is not None:
        user_ids = []
        with open(args.user_id_filepath, 'r') as infile:
            for line in infile:
                line = line.strip()
                if line != "":
                    user_ids.append(int(line))
            logger.info(f"Read {len(user_ids)} user ids from '{args.user_id_filepath}'.")
        user_id_list.extend(user_ids)
    user_id_set = set(user_id_list)
    if len(user_id_list) != len(user_id_set):
        logger.info(f"Removing duplicate user_ids.")
        user_id_list = list(user_id_set)
    logger.info(f"Identified {len(user_id_list)} user ids.")
    return user_id_list


def generate_rec_contexts(rdg, timestamp, user_ids):
    logger = logging.getLogger('cbrec.prediction.generate_rec_contexts')
    rdg.update_data()

    # start the database writer
    rdg.database_writer.start_process()
    try:
        logger.info(f"First prediction metadata id: {rdg.metadata_id_counter} (to generate: {len(user_ids)})")
        n_errors = 0
        for user_id in tqdm(user_ids, desc="Generating no-target predict recs"):
            metadata_id = rdg.metadata_id_counter
            rdg.metadata_id_counter += 1
            try:
                rc, _ = rdg.create_targetless_rec_context(metadata_id, timestamp, user_id, None)
                rdg.database_writer.save_rec_context(rc)
            except ValueError as ex:
                logger.error(ex)
                n_errors += 1
        logger.info(f"Last prediction metadata id: {rdg.metadata_id_counter - 1}")
        if n_errors > 0:
            logger.warning(f"Total of {n_errors} errors occurred generating rec contexts for prediction.")
    finally:
        rdg.database_writer.stop_process()
        logger.info("Database writer process stopped.")
    logger.info("Finished generating rec contexts.")


def identify_required_journals(config, source_user_id_list, candidate_usps, prediction_timestamp, data_manager):
    """
    Note: the make_text_features_predict.sh script will use the generated files to update the embedding database.
    """
    logger = logging.getLogger('cbrec.prediction.identify_required_journals')
    logger.info(f"Identifying journals for {len(candidate_usps)} candidate USPs.")
    
    jil = journalid.JournalIdLookup(config, data_manager)
    
    journal_oids = []
    n_not_enough = 0
    for candidate_usp in candidate_usps:
        journal_updates_before = jil.get_journal_updates_before(candidate_usp, prediction_timestamp + 1)
        if len(journal_updates_before) < 3:
            n_not_enough += 1
        else:
            journal_oids.extend(journal_updates_before)
    logger.info(f"{len(journal_oids)} journal ids identified for candidate USPs. ({n_not_enough} candidate USPs with <3 journals)")
    journal_oids = set(journal_oids)
    prediction_required_journal_ids_filepath = os.path.join(config.model_data_dir, 'predict_candidate_journal_oids.txt')
    with open(prediction_required_journal_ids_filepath, 'w') as outfile:
        for journal_oid in journal_oids:
            outfile.write(journal_oid + "\n")
    logger.info(f"Wrote {len(journal_oids)} journal ids to '{prediction_required_journal_ids_filepath}'.")
    
    journal_df = data_manager.get_filtered_journals() #sjournal_df[sjournal_df.is_nontrivial].sort_values(by=['user_id', 'site_id', 'created_at']).groupby(['user_id', 'site_id'])
    sjournal_df = journal_df[journal_df.user_id.isin(source_user_id_list)]
    # get source USPs from sjournal_df
    source_usps = list(set([(user_id, site_id) for user_id, site_id in zip(sjournal_df.user_id, sjournal_df.site_id)]))
    logger.info(f"Identified {len(source_usps)} source USPs from {len(source_user_id_list)} source users.")
    
    journal_oids = []
    n_not_enough = 0
    for source_usp in source_usps:
        journal_updates_before = jil.get_journal_updates_before(source_usp, prediction_timestamp + 1)
        if len(journal_updates_before) < 3:
            n_not_enough += 1
        else:
            journal_oids.extend(journal_updates_before)
    logger.info(f"{len(journal_oids)} journal ids identified for source USPs. ({n_not_enough} source USPs with <3 journals)")
    
    prediction_required_journal_ids_filepath = os.path.join(config.model_data_dir, 'predict_source_journal_oids.txt')
    with open(prediction_required_journal_ids_filepath, 'w') as outfile:
        for journal_oid in journal_oids:
            outfile.write(journal_oid + "\n")
    logger.info(f"Wrote {len(journal_oids)} journal ids to '{prediction_required_journal_ids_filepath}'.")
    
    if len(journal_oids) > 0:
        logger.info("Recommend running `sbatch -p amdsmall make_text_features_predict.sh` to ensure journals are present in embedding db.")
    else:
        logger.info("Seemingly no need to run make_text_features_predict.sh; no missing journal_oids.")
    
    
def main():
    logutils.set_up_logging()
    logger = logging.getLogger('cbrec.prediction.main')
    config = genconfig.Config()
    if not os.path.exists(config.model_data_dir):
        config = testutils.get_test_config()
        logger.info("Using test/debug configuration.")
    else:
        logger.info("Using standard configuration.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-name', dest='checkpoint_name', action='store', required=True)
    parser.add_argument('--user-ids', dest='user_id_list', action='store', default='')
    parser.add_argument('--user-ids-file', dest='user_id_filepath', action='store', default=None)
    parser.add_argument('--timestamp', dest='prediction_timestamp', action='store', required=True)
    parser.add_argument('--dry-run', dest='dry_run', action='store_true', default=False)
    parser.add_argument('--no-required-journals', dest='identify_required_journals', action='store_false', default=True)
    args = parser.parse_args()
    
    rdg = get_rec_data_generator(config, args.checkpoint_name)
    logger.info(f"Instantiated RecDataGenerator from pickle. (current_interaction_ind={rdg.current_interaction_ind})")

    active_user_ids = rdg.activity_manager.get_active_user_ids()
    eligible_user_ids = rdg.eligibility_manager.get_eligible_user_ids()
    eligible_active_user_ids = active_user_ids & eligible_user_ids
    logger.info(f"Identified {len(eligible_active_user_ids)} eligible active users as potential rec targets (from among {len(eligible_user_ids)} total eligible users).")
    
    if args.user_id_list == 'infer':
        # this enables coverage-based predictions
        selected_user_ids = config.rng.permutation(np.array(list(eligible_active_user_ids)))[:1000]
        logger.info(f"Making predictions for a subset of {len(selected_user_ids)} eligible active user_ids.")
        user_id_list = list(selected_user_ids)
    else:
        user_id_list = get_user_ids(args)
        logger.info(f"Making predictions for {len(user_id_list)} user ids.")
    
    # generate and save candidate usps for the eligible active users
    candidate_usps = []                                                                                                             
    for candidate_user_id in eligible_active_user_ids:
        for site_id in rdg.eligibility_manager.get_eligible_sites_from_user(candidate_user_id):
            candidate_usps.append((candidate_user_id, site_id))
    logger.info(f"Identified {len(candidate_usps)} USPs from {len(eligible_active_user_ids)} eligible active users.")
    candidate_usp_filepath = os.path.join(config.checkpoint_dir, f"{os.path.splitext(args.checkpoint_name)[0]}_prediction_usps.tsv")
    with open(candidate_usp_filepath, 'w') as outfile:
        for candidate_usp in candidate_usps:
            outfile.write(f"{candidate_usp[0]}\t{candidate_usp[1]}\n")
    logger.info(f"Wrote {len(candidate_usps)} candidate USPs to '{candidate_usp_filepath}'.")

    if args.prediction_timestamp == 'infer':
        prediction_timestamp = int(rdg.data_manager.ints_df.created_at.max())
        six_hours_ms = 1000 * 60 * 60 * 6
        prediction_timestamp += six_hours_ms
        logger.info(f"Inferred a prediction timestamp of {prediction_timestamp} ({datetime.utcfromtimestamp(prediction_timestamp / 1000).isoformat()}).")
    else:
        prediction_timestamp = int(args.prediction_timestamp)
    logger.info(f"Making predictions at time {datetime.utcfromtimestamp(prediction_timestamp / 1000)}")
    
    if args.identify_required_journals:
        identify_required_journals(config, user_id_list, candidate_usps, prediction_timestamp, rdg.data_manager)
        logger.info("Finished identified required journals.")
    
    if args.dry_run:
        logger.info("Dry run: not generating rec contexts.")
        return
    
    generate_rec_contexts(rdg, prediction_timestamp, user_id_list)
    
    logger.info("Finished.")
