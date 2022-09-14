"""
Runs the triple generation workflow.

usp = user/site pair
existing vs eligible vs active

"""
from . import genconfig
from . import network
from . import recentActivityCounter
from . import timeAwareDict
from . import feature_extraction
from . import data
from . import eligibility
from . import evaluation
from . import reccontext
from . import logutils
from . import testutils

import os
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse
import dateutil.parser
import pytz
import pickle
from glob import glob
#import cProfile

class RecDataGenerator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('cbrec.triple_generation.run')
        self.data_manager = data.DataManager(config)
        self.graph = network.UserGraph(config)
        self.activity_manager = recentActivityCounter.RecentActivityManager(config)
        self.eligibility_manager = eligibility.UserSitePairEligibilityManager(config)
        
        self.journal_dict = timeAwareDict.get_journal_ordered_dict(self.data_manager.get_filtered_journals())
        self.data_manager.clear_journal_data()  # we are finished with the journal data once journal_dict is constructed
    
        self.feature_generator = feature_extraction.FeatureGenerator(config, self.graph, self.activity_manager)
        self.database_writer = feature_extraction.DatabaseWriter(config)

        self.year_start_timestamps = [dateutil.parser.parse(f"{year}-01-01").replace(tzinfo=pytz.UTC).timestamp() * 1000 for year in [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]]
        self.in_generation_period = False
        self.in_test_period = False

        self.current_interaction_ind = 0  # used to track which interactions we've already processed
        self.metadata_id_counter = 0  # used to ensure the metadata_ids used remain globally unique
        
        # If should_generate_features is False, the database_writer will not be started and no rec contexts will be generated or saved.
        self.should_generate_features = config.should_generate_features
        self.logger.info("Instantiated new RecDataGenerator.")
    
    def __repr__(self):
        return f"RecDataGenerator(current_interaction_ind={self.current_interaction_ind}, metadata_id_counter={self.metadata_id_counter}, in_generation_period={self.in_generation_period}, in_test_period={self.in_test_period})"
    
    def __getstate__(self):
        # Copy the object's state
        state = self.__dict__.copy()
        # Remove unpicklable entries
        del state['data_manager']

        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        
        # recreate the data manager instance
        self.data_manager = data.DataManager(self.config)

    def create_checkpoint(self):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        # TODO consider including an optional key in the rdg to make it easier to keep track of what each checkpoint represents
        filename = f'rdg_{self.current_interaction_ind}_{self.metadata_id_counter}.pkl'
        checkpoint_filepath = os.path.join(self.config.checkpoint_dir, filename)
        with open(checkpoint_filepath, 'wb') as outfile:
            pickle.dump(self, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        # TODO it's possible the checkpoint should also save the current feature database and metadata filepaths... (could be handled directly in the WriterProcess?)
        self.logger.info(f"Created checkpoint at '{checkpoint_filepath}'.")

    def save_activity_manager(self):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_filepath = os.path.join(self.config.checkpoint_dir, f'activity_manager_predict.pkl')
        with open(checkpoint_filepath, 'wb') as outfile:
            pickle.dump(self.activity_manager, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"Saved activity manager at '{checkpoint_filepath}'.")

    def log_summary(self):
        self.logger.info(f"Summarizing RecDataGenerator: {str(self)}")
        self.logger.info(f"    journal_dict: {len(self.journal_dict)} entries")

    def update_data(self):
        """
        Set metadata_id and update filter_data. Should be called before running rec data generation or if the config is updated.
        """
        # get the current metadata id, which will be 0 if the database is empty
        curr_metadata_id = self.metadata_id_counter
        self.metadata_id_counter = self.database_writer.get_current_metadata_id() + 1
        self.logger.info(f"Setting metadata_id to {self.metadata_id_counter} (was {curr_metadata_id}).")

        # verify that all available journals are loaded into the journal_dict
        if self.data_manager.get_filtered_journals() is not None:
            filtered_journal_df = self.data_manager.get_filtered_journals()
            timeAwareDict.verify_journal_ordered_dict(self.journal_dict, filtered_journal_df)
            self.data_manager.clear_journal_data()  # we are finished with the journal data once journal_dict is constructed
            self.logger.info(f"Cleared data_manager journal data after verifying journal_dict (size = {len(self.journal_dict)}).")
        else:
            self.logger.info(f"Not verifying journal_dict, as data_manager journal data is not loaded. (Likely because it was already verified in __init__.)")

    def replace_config(self, config, refresh_data=True):
        """
        Hacky function to make some desired mutations possible.

        This will NOT do everything you expect, and most config settings will be unaffected.
        """
        self.config = config
        self.should_generate_features = config.should_generate_features
        if refresh_data:
            self.data_manager = data.DataManager(config)

    def generate_rec_data(self):
        """
        The fundamental purpose of this function is to use any available data to compute train and test recommendation contexts.
        """
        self.update_data()

        # start the database writer
        if self.should_generate_features:
            self.database_writer.start_process()
        else:
            self.logger.info("Not starting database_writer: should_generate_features is False.")

        # note: assume the ints_df is sorted
        df = self.data_manager.ints_df[self.data_manager.ints_df.created_at >= self.config.first_int_timestamp]
        i = 0
        is_skipping = False
        for int_created_at, int_user_id, int_site_id, interaction_type in tqdm(
                zip(df.created_at, df.user_id, df.site_id, df.interaction_type), 
                total=len(df), desc='Processing interactions'):
            if i < self.current_interaction_ind:
                i += 1
                is_skipping = True
                continue
            elif i > self.current_interaction_ind:
                raise ValueError(f"State inconsistency; expected invariant i <= self.current_interaction_ind, instead found i={i} > self.current_interaction_ind={self.current_interaction_ind}")
            elif is_skipping:
                self.logger.info(f"Was skipping to catch up with current counter. Caught up at index {i}.")
                is_skipping = False
            done_processing = self.process_interaction(int_created_at, int_user_id, int_site_id, interaction_type)
            if done_processing:
                self.logger.info(f"Finished processed after interaction {i}.")
                break
            i += 1
            self.current_interaction_ind += 1

            # TODO check for SIGKILL and gracefully terminate loop processing, perhaps making a checkpoint?

        if self.should_generate_features:
            self.logger.info("Finished processing; waiting for database writer.")
            self.database_writer.stop_process()
        self.logger.info("Finished generating rec data; creating checkpoint.")
        # TODO is there something that can be done here to save memory? Maybe clearing the database_writer's cache...
        #while len(self.database_writer.feature_arr_cache) > 0: 
        #    self.database_writer.feature_arr_cache.popitem()
        #self.logger.info("Cleared database writer's feature cache.")
        self.create_checkpoint()  # to make subsequent predictions easier, make a checkpoint after processing all interactions

    def check_for_yearly_progress(self, int_created_at):
        # check for new year, in order to log progress
        if len(self.year_start_timestamps) > 0 and int_created_at >= self.year_start_timestamps[0]:
            new_year = self.year_start_timestamps.pop(0)
            if not self.check_for_yearly_progress(int_created_at):  # do a recursive check for other updates
                self.logger.info(f"Processing first timestamp in {datetime.utcfromtimestamp(new_year / 1000).strftime('%Y-%m')} at {datetime.now()}")
            return True  # this is the first reaction we've seen in the next year
        return False

    def check_for_progress_point(self, int_created_at):
        self.check_for_yearly_progress(int_created_at)
        if not self.in_generation_period and int_created_at >= self.config.generation_start_timestamp:
            self.logger.info("Started processing interactions in generation period.")
            self.in_generation_period = True
            self.create_checkpoint()  # make a checkpoint right before we start processing in the generation period
        if not self.in_test_period and int_created_at >= self.config.test_generation_start_timestamp:
            self.logger.info("Started processing interactions in test period.")
            self.in_test_period = True
            
            # track users that were existing or eligible in the train period
            self.eligibility_manager.update_training_period_ids()
            
            # create a set of coverage users
            self.create_coverage_users(int_created_at)

            # clear the cache of all training arrays; will be testing period now
            self.database_writer.feature_arr_cache.clear_cache()
            # decrease the size of the cache, as the testing period arrays are much larger (~12000 X n_features)
            # FIXME is this reasonable?
            self.database_writer.feature_arr_cache.maxsize = 10
            
            # make a checkpoint right before we start processing interactions in the test period
            self.create_checkpoint()  
            
    def create_coverage_users(self, timestamp):
        active_user_ids = self.activity_manager.get_active_user_ids()
        eligible_user_ids = self.eligibility_manager.get_eligible_user_ids()
        eligible_active_user_ids = active_user_ids & eligible_user_ids
        self.logger.info(f"Coverage: Identified {len(eligible_active_user_ids)} eligible active users as potential rec targets (from among {len(eligible_user_ids)} total eligible users).")
        n_coverage_users = min(self.config.n_coverage_users, len(eligible_active_user_ids))
        if n_coverage_users < self.config.n_coverage_users:
            self.logger.warn(f"Coverage: Expected at least {self.config.n_coverage_users} eligible users at timestamp {timestamp} for coverage predictions, but found only {n_coverage_users}.")
        selected_user_ids = self.config.rng.permutation(np.array(list(eligible_active_user_ids)))[:n_coverage_users]
        # temporarily expand the feature_generator's cache so that we don't end up with hordes of duplicate ids/duplicate work
        # (this is the special case where the timestamp doesn't change, so the cache actually works)
        original_cache_size = self.feature_generator.feature_arr_cache.maxsize
        self.feature_generator.feature_arr_cache.maxsize = len(eligible_active_user_ids)
        # set the timestamp for which recs are generated
        # we set this to a time in the future to better replicate the actual recommendation conditions
        timestamp += 1000 * 60 * 60 * 12  # 12 hours
        self.logger.info(f"Coverage: generating recommendations at timestamp {timestamp} ({datetime.utcfromtimestamp(timestamp / 1000).isoformat()})")
        # generate non-target RecContexts for the selected user_ids
        for user_id in tqdm(selected_user_ids, desc="Generating no-target predict recs"):
            metadata_id = self.metadata_id_counter
            self.metadata_id_counter += 1
            rc, _ = self.create_targetless_rec_context(metadata_id, timestamp, user_id, use_cache=True)
            self.database_writer.save_rec_context(rc)
        self.feature_generator.feature_arr_cache.maxsize = original_cache_size
        self.feature_generator.feature_arr_cache.clear_cache()
        self.logger.info("Coverage: finished generating coverage predictions.")

    def create_targetless_rec_context(self, metadata_id, created_at, user_id, save_scores=False, use_cache=None):
        """
            :use_cache - propagates to RecContext.generate_features().  If none (default), use the default implementation.

            :returns
                RecContext - the newly created RecContext
                Scorer - deprecated, now returns only None
        """
        rc = reccontext.RecContext(self.config, metadata_id, created_at, user_id, None)
        rc.identify_eligibility(self.eligibility_manager)
        if not rc.is_initiation_eligible:
            raise ValueError(f"User {user_id} at time {created_at} (metadata id = {metadata_id}) is not eligible. (source_user_is_eligible={rc.md['source_user_is_eligible']})")
        rc.identify_usps(self.activity_manager, self.graph, self.eligibility_manager)
        rc.generate_features(self.feature_generator, use_cache=use_cache)
        rc.record_journal_ids(self.eligibility_manager)
        assert rc.is_test_period
        #scorer = evaluation.BaselineScorer(self.config, rc, save_scores=save_scores)
        #scorer.compute_baselines(self.activity_manager)
        return rc, None

    def process_interaction(self, int_created_at, int_user_id, int_site_id, interaction_type):
        """
        :returns -- True if done processing, false if additional interactions will be accepted.
        """
        self.check_for_progress_point(int_created_at)

        # update journal activity
        new_journals = self.journal_dict.get_payloads_before_timestamp(int_created_at)
        if len(new_journals) >= 10000:
            self.logger.warn(f"Identified {len(new_journals)} journal updates at time {int_created_at} ({datetime.utcfromtimestamp(int_created_at / 1000).replace(tzinfo=pytz.UTC)}).")
        for new_journal in new_journals:
            journal_user_id, journal_site_id, journal_oid, journal_created_at = new_journal
            self.activity_manager.add_interaction('journal_user', journal_user_id, journal_created_at)
            self.activity_manager.add_interaction('journal_site', journal_site_id, journal_created_at)
            is_new_existing, _ = self.eligibility_manager.add_recent_journal_update(journal_user_id, journal_site_id, journal_oid)
            # note: this will update the sets of existing and active users
            if is_new_existing:
                # MAY need to update the graph, as a new user is eligible on this site
                # (we may have never seen this site before)
                # so, get all users who have previously interacted with this site
                prev_user_ids = self.eligibility_manager.get_existing_site_ints(journal_site_id)
                for prev_user_id in prev_user_ids:
                    # add edge prev_user_id -> journal_user_id
                    self.graph.add_edge(prev_user_id, journal_user_id)

        # is_initiation if this is the first time this user_id has interacted with this site_id
        is_initiation = self.eligibility_manager.is_initiation(int_user_id, int_site_id)
                    
        # update activity counters to the current moment
        self.activity_manager.update_counts(int_created_at)
        
        if self.config.generation_stop_timestamp is not None and int_created_at > self.config.generation_stop_timestamp:
            # log data about eligible users
            active_user_ids = self.activity_manager.get_active_user_ids()
            eligible_user_ids = self.eligibility_manager.get_eligible_user_ids()
            eligible_active_user_ids = active_user_ids & eligible_user_ids
            self.logger.info(f"At end of the test period, there are {len(eligible_active_user_ids)} eligible active users.")
            self.logger.info("Note: if end-of-period predictions are desired, see the predict.py script.")
            
            # return true to indicate that we are done processing interactions
            return True
        elif int_created_at >= self.config.generation_start_timestamp and is_initiation and self.should_generate_features:
            # for initiations after some time period, we generate features
            # we also need the source user to be eligible, and at least one user on the receiving site to be eligible
            # TODO this should be a config setting
            metadata_id = self.metadata_id_counter
            self.metadata_id_counter += 1
            rc = reccontext.RecContext(self.config, metadata_id, int_created_at, int_user_id, int_site_id)
            rc.identify_eligibility(self.eligibility_manager)
            if rc.is_initiation_eligible:
                rc.identify_usps(self.activity_manager, self.graph, self.eligibility_manager)
                rc.generate_features(self.feature_generator)
                rc.record_journal_ids(self.eligibility_manager)
                #if rc.is_test_period:
                #    scorer = evaluation.BaselineScorer(self.config, rc)
                #    scorer.compute_baselines(self.activity_manager)
            self.database_writer.save_rec_context(rc)

        # update network
        if is_initiation:
            self.eligibility_manager.add_site_int(int_user_id, int_site_id)
            self.activity_manager.add_interaction('initiation_site', int_site_id, int_created_at)
            # who has int_user_id interacted with in the graph?
            # any user who has previously authored an update on this site
            curr_existing_users_on_site = self.eligibility_manager.get_existing_users_on_site(int_site_id)
            for curr_user_id in curr_existing_users_on_site:
                # add edge int_user_id -> curr_user_id
                self.graph.add_edge(int_user_id, curr_user_id)

        # update activity counters
        self.activity_manager.add_interaction(interaction_type, int_user_id, int_created_at)
        return False


def get_available_checkpoints(checkpoint_dir):
    """
    :returns - a list of (current_interaction_ind, metadata_id, checkpoint_filepath) tuples,
    sorted so the last element has the highest interaction ind
    """
    available_checkpoints = []

    checkpoint_filepaths = glob(os.path.join(checkpoint_dir, 'rdg_*.pkl'))
    for filepath in checkpoint_filepaths:
        filename = os.path.basename(filepath)
        filename_noext = os.path.splitext(filename)[0]
        _, current_interaction_ind, metadata_id = filename_noext.split("_")
        current_interaction_ind = int(current_interaction_ind)
        metadata_id = int(metadata_id)
        available_checkpoints.append((current_interaction_ind, metadata_id, filepath))
    available_checkpoints.sort()
    return available_checkpoints


def main():
    logutils.set_up_logging()
    logger = logging.getLogger('cbrec.triple_generation.main')
    config = genconfig.Config()
    if not os.path.exists(config.model_data_dir):
        config = testutils.get_test_config()
        logger.info("Using test/debug configuration.")
    else:
        logger.info("Using standard configuration.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--from-checkpoint', dest='from_checkpoint', action='store', default=None)
    parser.add_argument('--from-recent-checkpoint', dest='from_recent_checkpoint', action='store_true', default=False)
    parser.add_argument('--list-checkpoints', dest='list_checkpoints', action='store_true', default=False)
    parser.add_argument('--summarize-checkpoint', dest='summarize_checkpoint', action='store', default=None)
    parser.add_argument('--interactive', dest='interactive', action='store_true', default=False)
    parser.add_argument('--load-only', dest='is_load_only', action='store_true', default=False)
    parser.add_argument('--no-features', dest='skip_feature_generation', action='store_true', default=False)
    args = parser.parse_args()
    
    if args.list_checkpoints:
        available_checkpoints = get_available_checkpoints(config.checkpoint_dir)
        for ap in available_checkpoints:
            logger.info(f"Found checkpoint {ap[0]} {ap[1]} at '{ap[2]}'.")
        checkpoint_filepath = available_checkpoints[-1][2]
        logger.info(f"Would load {checkpoint_filepath} if resuming.")
        return

    if args.summarize_checkpoint is not None:
        if args.summarize_checkpoint == 'infer':
            available_checkpoints = get_available_checkpoints(config.checkpoint_dir)
            checkpoint_filepath = available_checkpoints[-1][2]
            logger.info(f"Inferred checkpoint '{checkpoint_filepath}'.")
            assert os.path.exists(checkpoint_filepath), "Checkpoint unexpectedly missing."
        elif os.path.exists(args.summarize_checkpoint):
            checkpoint_filepath = args.summarize_checkpoint
        else:
            checkpoint_filepath = os.path.join(config.checkpoint_dir, args.summarize_checkpoint)
            if not os.path.exists(checkpoint_filepath):
                logger.error(f"File '{checkpoint_filepath}' does not exist.")
                return
        with open(checkpoint_filepath, 'rb') as infile:
            s = datetime.now()
            rdg = pickle.load(infile)
        logger.info(f"Loaded rdg {str(rdg)} in {datetime.now() - s}.")
        rdg.log_summary()
        if args.interactive:
            logger.info("Starting interpreter; <rdg> instance is loaded.")
            import IPython
            IPython.embed()
        return

    start_from_checkpoint = args.from_recent_checkpoint or args.from_checkpoint is not None
    if start_from_checkpoint:
        # TODO remove me, but for now we force removal of any existing data in standard config as well
        # should only do this if loading from checkpoint
        #os.remove(config.feature_db_filepath)
        #os.remove(config.metadata_filepath)
        # the two debug checkpoints are: rdg_0_0 (before generation period) and rdg_88805_14642 (before test period)
        #checkpoint_filepath = os.path.join(config.checkpoint_dir, 'rdg_88805_14642.pkl')
        #checkpoint_filepath = os.path.join(config.checkpoint_dir, 'rdg_23453504_396530.pkl')
        available_checkpoints = get_available_checkpoints(config.checkpoint_dir)
        for ap in available_checkpoints:
            logger.info(f"Found checkpoint {ap[0]} {ap[1]} at '{ap[2]}'.")
        if args.from_recent_checkpoint:
            checkpoint_filepath = available_checkpoints[-1][2]
        else:
            checkpoint_filepath = os.path.join(config.checkpoint_dir, args.from_checkpoint)
            assert os.path.exists(checkpoint_filepath)
        logger.info(f"Loading checkpoint from filepath '{checkpoint_filepath}'.")
        with open(checkpoint_filepath, 'rb') as infile:
            rdg = pickle.load(infile)
            logger.info(f"Instantiated RecDataGenerator from pickle. (current_interaction_ind={rdg.current_interaction_ind})")
    else:
        rdg = RecDataGenerator(config)
    if args.is_load_only:
        logger.info("Load only; terminating.")
        return
    
    if args.skip_feature_generation:
        rdg.should_generate_features = False
        logger.info("Skipping feature generation; setting should_generate_features to false.")
    else:
        rdg.should_generate_features = True
    
    # run the generation process
    rdg.generate_rec_data()

    logger.info("Finished.")
    
    
if __name__ == '__main__':
    main()
