
import cbrec.genconfig
import cbrec.data

import os
from datetime import datetime
import pytz

def main():
    s = datetime.now()
    config = cbrec.genconfig.Config()
    dm = cbrec.data.DataManager(config)

    git_root_dir = cbrec.genconfig.get_git_root_dir()
    input_test_data_dir = os.path.join(git_root_dir, 'src/test/data/input')

    # the test period starts on Jan 1 2018, so we take for testing a small period around the inflection point
    start_timestamp = datetime.fromisoformat('2017-12-20').replace(tzinfo=pytz.UTC).timestamp() * 1000
    stop_timestamp = datetime.fromisoformat('2018-01-10').replace(tzinfo=pytz.UTC).timestamp() * 1000


    dm.ints_df[(dm.ints_df.created_at >= start_timestamp)&(dm.ints_df.created_at <= stop_timestamp)].reset_index().to_feather(os.path.join(input_test_data_dir, 'ints_df.feather'))

    journal_df = dm.journal_df
    journal_df[(journal_df.created_at >=  start_timestamp)&(journal_df.created_at <= stop_timestamp)].reset_index().to_feather(os.path.join(input_test_data_dir, 'journal_metadata.feather'))

    print(f"Finished making test data in {datetime.now() - s}.")

    
if __name__ == '__main__':
    main()
