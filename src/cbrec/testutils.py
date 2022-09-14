from . import genconfig

import os

def get_test_config():
    config = genconfig.TestConfig()
    # as a special test configuration setting, remove existing paths
    #if os.path.exists(config.feature_db_filepath): 
    #    os.remove(config.feature_db_filepath)
    #if os.path.exists(config.metadata_filepath): 
    #    os.remove(config.metadata_filepath)
    return config