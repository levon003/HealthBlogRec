
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np

class DataManager:
    def __init__(self, config, load_journals=True, load_ints=True):
        self.logger = logging.getLogger("cbrec.data.DataManager")
        self.config = config
        
        self.user_site_df = None
        self.valid_user_ids = None
        self.valid_site_ids = None
        self.ints_df = None
        self.journal_df = None
        self.filtered_journal_df = None
        
        self.__load_data(load_journals=load_journals, load_ints=load_ints)
        self.trim_data()
        self.get_filtered_journals()

    def __load_data(self, load_journals=True, load_ints=True):
        # load the list of valid user/site pairs
        s = datetime.now()
        self.user_site_df = pd.read_csv(os.path.join(self.config.model_data_dir, 'user_site_df.csv'))
        self.valid_user_ids = set(self.user_site_df.user_id)
        self.valid_site_ids = set(self.user_site_df.site_id)
        self.logger.info(f"Read {len(self.user_site_df)} user_site_df rows ({len(self.valid_user_ids)} unique users, {len(self.valid_site_ids)} unique sites) in {datetime.now() - s}.")

        # read interactions dataframe
        if load_ints:
            s = datetime.now()
            ints_trimmed_filepath = os.path.join(self.config.model_data_dir, 'ints_df_trimmed.feather')
            if os.path.exists(ints_trimmed_filepath):
                self.ints_df = pd.read_feather(ints_trimmed_filepath)
                self.logger.debug("Reading from existing trimmed ints_df file.")
            else:
                self.ints_df = pd.read_feather(os.path.join(self.config.model_data_dir, 'ints_df.feather'))
            self.logger.info(f"Read {len(self.ints_df)} ints_df rows ({len(set(self.ints_df.user_id))} unique users) in {datetime.now() - s}.")

        # load the journal dataframe
        if load_journals:
            s = datetime.now()
            journal_metadata_filepath = os.path.join(self.config.journal_metadata_dir, "journal_metadata.feather")
            self.journal_df = pd.read_feather(journal_metadata_filepath)
            self.logger.info(f"Read {len(self.journal_df)} journal_df rows in {datetime.now() - s}.")

            if self.config.debug_mode:
                self.logger.info("Debug: Duplicating rows to increase eligibility during test period.")
                start_timestamp = np.min(self.journal_df.created_at) - 1000  # 1 second before the first journal in the period
                user_id_site_map = {user_id: site_id for user_id, site_id in zip(self.journal_df.user_id, self.journal_df.site_id)}
                new_columns = [{'created_at': start_timestamp, 'journal_oid': str(user_id) + "debug", 'published_at': start_timestamp, 'site_id': user_id_site_map[user_id],
           'updated_at': start_timestamp, 'user_id': user_id, 'site_index': -1, 'is_nontrivial': True} for user_id in set(self.journal_df.user_id)]
                debug_journal_df = pd.DataFrame(new_columns)
                self.journal_df = pd.concat([debug_journal_df, self.journal_df], ignore_index=True)
                self.logger.info(f"Debug: Added {len(debug_journal_df)} rows for a total of {len(self.journal_df)} rows.")

    def trim_data(self):
        """
        Trim the data.
        
        TODO Determine if we need to filter out based on the user_site_df, i.e. if there are invalid users in the journal_df who ought to be removed. I think this might actually be a subtle and bad problem: we need the journals and interactions from the non-eligible authors in order to ensure the graph stays up to date, but we need to note allow invalid authors to be eligible. Is that true at all?  Need to check on this....   Seems like we need to remove the spam/manual removals, but keep everyone with less activity.
        Later edit: I believe this TODO to have been resolved...
        """
        # trim out journal updates that are trivial (short or machine-generated)
        if self.journal_df is not None:
            self.journal_df = self.journal_df[self.journal_df.is_nontrivial]
        
            # trim out journal updates with invalid dates
            self.logger.info(f"Keeping journals published between {self.config.invalid_start_date.isoformat()} and {self.config.invalid_end_date.isoformat()}.")
            self.journal_df = self.journal_df[(self.journal_df.published_at >= self.config.invalid_start_timestamp)&(self.journal_df.published_at <= self.config.invalid_end_timestamp)]
            self.logger.info(f"New journal_df count: {len(self.journal_df)}")
        
        # trim out interactions with invalid dates
        if self.ints_df is not None:
            ints_trimmed_filepath = os.path.join(self.config.model_data_dir, 'ints_df_trimmed.feather')
            if not os.path.exists(ints_trimmed_filepath):
                self.logger.info("Computing and caching trimmed ints_df.")
                
                old_ints_df_size = len(self.ints_df)
                self.ints_df = self.ints_df[(self.ints_df.created_at>=self.config.invalid_start_timestamp)&(self.ints_df.created_at<=self.config.invalid_end_timestamp)].reset_index(drop=True)
                self.logger.info(f"New ints_df count: {len(self.ints_df)} / {old_ints_df_size}")

                # manage amp data
                amp_timestamps_to_adjust = self.ints_df.interaction_type == 'amp'  # original pre-reactions amps have int_type == 'amp'
                n_adjusted = np.sum(amp_timestamps_to_adjust)
                if n_adjusted > 0:
                    # see notebook/model_data/AmpTimestampFix.ipynb for the source of this data
                    # which contains the time elapsed between Reaction creation date (createdAt) and Journal publication (publishedAt)
                    # note that this causes some weirdness when the createdAt date of the journals != publishedAt, which happens about 20% of the time
                    # (due to a bug in the amp extraction code in cbcore, unless that's been fixed)
                    reaction_times_filepath = os.path.join(self.config.model_data_dir, 'reaction_ms_since_journal.npy')
                    with open(reaction_times_filepath, 'rb') as infile:
                        reaction_ms_since_journal = np.load(infile)
                    assert len(reaction_ms_since_journal) > 0
                    rng = self.config.rng

                    # add noise to existing amp timestamps
                    new_created_at = self.ints_df.loc[amp_timestamps_to_adjust].created_at.map(lambda ca: ca + rng.choice(reaction_ms_since_journal))
                    self.ints_df.loc[amp_timestamps_to_adjust, 'created_at'] = new_created_at
                    self.logger.info(f"Added noise to created_at timestamps for {n_adjusted} amp interactions.")

                # normalize labels
                self.ints_df.loc[self.ints_df.interaction_type.str.startswith('amp'), 'interaction_type'] = 'amp'

                # verify ints_df is in sorted order
                self.ints_df.sort_values(by='created_at', ascending=True, inplace=True)
                
                # reset index
                self.ints_df.reset_index(inplace=True, drop=True)
                self.logger.info("Trimmed and sorted ints_df.")
                
                # cache the trimmed ints_df to a file
                self.ints_df.to_feather(ints_trimmed_filepath)
                self.logger.info(f"Saved to '{ints_trimmed_filepath}'.")
            else:
                self.logger.info("Using cached version of ints_df trimmed.")

    def get_filtered_journals(self):
        if self.filtered_journal_df is not None:
            return self.filtered_journal_df
        if self.journal_df is None:
            # only populate filtered_journal_df if the journal_df is available
            return None
        # use only journal updates authored by users who will become eligible (edit: no longer true, now it's any non-spam user who will become existing, not necessarily eligible)
        # note this includes updates in (user_id, site_id) pairs that aren't in valid_usps
        # but we still want to record this activity for feature extraction reasons
        s = datetime.now()
        self.filtered_journal_df = self.journal_df[self.journal_df.user_id.isin(self.valid_user_ids)]
        self.logger.info(f"Identified {len(self.filtered_journal_df)} / {len(self.journal_df)} journal updates from valid authors in {datetime.now() - s}.")
        return self.filtered_journal_df

    def clear_journal_data(self):
        """
        Sets journal_df and filtered_journal_df to None, which will allow Python to free that memory for other objects.
        """
        self.journal_df = None
        self.filtered_journal_df = None
