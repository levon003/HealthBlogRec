
import os
import numpy as np
from datetime import datetime
import pytz
from numpy.random import default_rng
import subprocess


def get_git_root_dir():
    res = subprocess.run(["git", "rev-parse", "--show-toplevel"], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    path = res.stdout.decode('utf-8').strip()
    return path


class Config():
    def __init__(self, config_key="default"):
        self.config_key = config_key
        self.debug_mode = False

        self.torch_experiments_dir = '/home/lana/shared/caringbridge/data/projects/recsys-peer-match/torch_experiments'
        self.model_data_dir = '/home/lana/shared/caringbridge/data/projects/recsys-peer-match/model_data'
        self.journal_metadata_dir = "/home/lana/shared/caringbridge/data/derived/journal_metadata"
        self.feature_data_dir = '/home/lana/shared/caringbridge/data/projects/recsys-peer-match/feature_data'
        self.feature_db_filepath = os.path.join(self.feature_data_dir, 'feature.sqlite')
        self.metadata_filepath = os.path.join(self.feature_data_dir, 'metadata.ndjson')
        self.checkpoint_dir = os.path.join(self.feature_data_dir, 'checkpoints')
        self.coverage_stats_dir = os.path.join(self.feature_data_dir, 'coverage')
        
        self.raw_text_db_filepath = '/home/lana/shared/caringbridge/data/derived/sqlite/journalText.sqlite'
        self.text_feature_db_filepath = os.path.join(self.feature_data_dir, 'text_feature.sqlite')

        self.should_clear_metadata_filepath = False
        self.should_clear_feature_db_filepath = False
        self.use_writer_process = True

        self.feature_arr_cache_size = 5000  # number of numpy arrays to cache for potential reuse (cache tracks feature_ids for potential reuse)
        self.feature_db_max_commit_size = 500000
        # TODO possible config self.feature_gen_max_queue_size = 10000000, that limits the total size of the IO backlog that can build up (and blocks until the backlog is 0 once exceeded)

        self.activity_count_duration_ms = 1000 * 60 * 60 * 24 * 7  # 1 week in ms
        self.interaction_count_duration_ms = 1000 * 60 * 60 * 24 * 365  # 1 year in ms
    
        self.invalid_start_date = datetime.fromisoformat('2005-01-01').replace(tzinfo=pytz.UTC)
        self.invalid_end_date = datetime.fromisoformat('2022-01-01').replace(tzinfo=pytz.UTC)
        self.invalid_start_timestamp = self.invalid_start_date.timestamp() * 1000
        self.invalid_end_timestamp = self.invalid_end_date.timestamp() * 1000
        
        # burn-in period for when to start tracking activity
        self.activity_start_date = datetime.fromisoformat('2013-12-01').replace(tzinfo=pytz.UTC)
        self.activity_start_timestamp = self.activity_start_date.timestamp() * 1000
        
        # the date from which to start processing interactions
        # interactions before this date are ignored in terms of both graph and activity
        self.first_int_timestamp = datetime.fromisoformat('2010-01-01').replace(tzinfo=pytz.UTC).timestamp() * 1000
        #self.first_int_timestamp = self.invalid_start_date  # TODO should use the start
        
        self.generation_start_timestamp = datetime.fromisoformat('2014-01-01').replace(tzinfo=pytz.UTC).timestamp() * 1000
        self.test_generation_start_timestamp = datetime.fromisoformat('2021-01-01').replace(tzinfo=pytz.UTC).timestamp() * 1000
        #self.generation_stop_timestamp = None  # generate until we run out of data
        self.generation_stop_timestamp = datetime.fromisoformat('2022-01-01').replace(tzinfo=pytz.UTC).timestamp() * 1000
        
        self.rng = default_rng(872)  # rng seed chosen by a post-doctoral researcher of mathematics, to ensure effective seeding

        self.ms_per_hour = 1000 * 60 * 60
        
        # max number of journal updates that one would expect in an eligible user to be available for feature extraction
        self.journal_update_memory = 3

        self.n_coverage_users = 1000  # the number of users to generate predictions for at the end of the training period
        
        self.n_users_to_predict = 1000  # I think this is deprecated now; not sure
        self.number_of_initiation = None
        # defined by the implementation in feature_extraction
        # saved with the config so that even if this number changes in the future, can still get correct feature array shape from the config used to generate those features
        self.user_feature_count = 3 + 9
        self.user_pair_feature_count = 3
        self.text_feature_count = 768
        
        self.pool_text_feature = 0  # 0 for average/mean pooling, 1 for max pooling, 2 for concatenating
        self.should_generate_features = True
        

    
class TestConfig(Config):
    def __init__(self, config_key="default_debug"):
        super().__init__(config_key=config_key)
        self.debug_mode = True

        git_root_dir = get_git_root_dir()
        test_data_dir = os.path.join(git_root_dir, 'src/test/data')

        self.model_data_dir = os.path.join(test_data_dir, 'input')
        self.journal_metadata_dir = os.path.join(test_data_dir, 'input')
        self.feature_data_dir = os.path.join(test_data_dir, 'output')
        self.feature_db_filepath = os.path.join(self.feature_data_dir, 'feature.sqlite')
        self.metadata_filepath = os.path.join(self.feature_data_dir, 'metadata.ndjson')
        self.checkpoint_dir = os.path.join(self.feature_data_dir, 'checkpoints')
        self.coverage_stats_dir = os.path.join(self.feature_data_dir, 'coverage')

        self.should_clear_metadata_filepath = False
        self.should_clear_feature_db_filepath = False

        # align the stop of generation with the end of the test data, to ensure the expected cleanup happens
        self.generation_start_timestamp = datetime.fromisoformat('2017-01-01').replace(tzinfo=pytz.UTC).timestamp() * 1000
        self.test_generation_start_timestamp = datetime.fromisoformat('2018-01-01').replace(tzinfo=pytz.UTC).timestamp() * 1000
        self.generation_stop_timestamp = datetime.fromisoformat('2018-01-09').replace(tzinfo=pytz.UTC).timestamp() * 1000

        self.n_users_to_predict = 200


