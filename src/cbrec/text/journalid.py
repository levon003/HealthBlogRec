
import logging
import bisect
from datetime import datetime
from tqdm import tqdm

class JournalIdLookup:
    """
    This class enables fast lookup of journal_ids given a timestamp.
    
    Uses published_at date in the journals dataframe.
    """
    def __init__(self, config, data_manager):
        self.logger = logging.getLogger("cbrec.text.journalid.JournalIdLookup")
        self.config = config
        s = datetime.now()
        journal_df = data_manager.get_filtered_journals().sort_values(by=['user_id', 'site_id', 'published_at'])
        self.logger.info(f"Sorted {len(journal_df)} journals in {datetime.now() - s}.")
        self.journal_id_set = set(journal_df.journal_oid)
        
        self.usp_journal_timestamp_map = {}
        self.usp_journal_id_map = {}
        
        current_usp = None
        current_timestamp_list = []
        current_journal_id_list = []
        for row in tqdm(journal_df.itertuples(), total=len(journal_df), desc="JournalIdLookup map construction"):
            usp = (row.user_id, row.site_id)
            if usp != current_usp:
                current_usp = usp
                current_timestamp_list = []
                current_journal_id_list = []
                self.usp_journal_timestamp_map[usp] = current_timestamp_list
                self.usp_journal_id_map[usp] = current_journal_id_list
            current_timestamp_list.append(row.published_at)
            current_journal_id_list.append(row.journal_oid)
        self.logger.info(f"Translated {len(journal_df)} journals into a map of {len(self.usp_journal_id_map)} USPs.")
    
    
    def get_journal_updates_before(self, usp, timestamp):
        if usp in self.usp_journal_timestamp_map:
            timestamp_list = self.usp_journal_timestamp_map[usp]
            end_ind = bisect.bisect_right(timestamp_list, timestamp)
            if end_ind is None:
                return []
            start_ind = max(end_ind - self.config.journal_update_memory, 0)
            journal_id_list = self.usp_journal_id_map[usp]
            journal_ids = journal_id_list[start_ind:end_ind]
            return journal_ids
        else:
            return []
        