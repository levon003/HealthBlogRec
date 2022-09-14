"""
Data structure for efficiently retrieving journals that occurred before some specific timestamp.

get_usp_eligibility_dicts is now deprecated.
"""


import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict


class TimeAwareDict:
    """
    TimeAwareDict is mapping of integer timestamp to a payload, 
    and provides get_payloads_before_timestamp to retrieve a list of payloads before a particular timstamp.

    To construct, insert_payload must be called in monotonically-increasing timestamp order.
    """
    def __init__(self):
        self.od = OrderedDict()
        
    def __len__(self):
        return len(self.od)
    
    def insert_payload(self, timestamp, payload):
        if timestamp in self.od:
            self.od[timestamp].append(payload)
        else:
            self.od[timestamp] = [payload,]
    
    def get_payloads_at_timestamp(self, timestamp):
        payloads = []
        if timestamp in self.od:
            payloads = self.od[timestamp]
        return payloads
    
    def get_payloads_before_timestamp(self, target_timestamp):
        # update up till the target_timestamp
        all_payloads = []
        while len(self.od) > 0:
            if next(iter(self.od)) < target_timestamp:
                timestamp, payloads = self.od.popitem(last=False)
                all_payloads.extend(payloads)
            else:
                break
        return all_payloads

    def peek_first_timestamp(self):
        if len(self.od) == 0:
            return None
        return next(iter(self.od))

    def peek_last_timestamp(self):
        if len(self.od) == 0:
            return None
        return next(reversed(self.od))

    
def get_journal_ordered_dict(filtered_journal_df):
    logger = logging.getLogger("cbrec.timeAwareDict.get_journal_ordered_dict")
    # map of published_at -> (user_id, site_id, journal_oid)
    s = datetime.now()
    journal_dict = TimeAwareDict()
    filtered_journal_df = filtered_journal_df.sort_values(by='published_at', ascending=True)
    for row in tqdm(filtered_journal_df.itertuples(), total=len(filtered_journal_df), desc="Building journal TimeAwareDict"):
        payload = (row.user_id, row.site_id, row.journal_oid, row.published_at)
        journal_dict.insert_payload(row.published_at, payload)
    logger.info(f"Created filtered journal odict with {len(journal_dict)} distinct published_at values in {datetime.now() - s}.")
    return journal_dict


def verify_journal_ordered_dict(journal_dict, filtered_journal_df):
    """
    Adds entries to journal_dict to match filtered_journal_df entries.

    Only looks for new entries after the last timestamp already present in the journal_dict; won't catch middle insertions.
    """
    logger = logging.getLogger("cbrec.timeAwareDict.verify_journal_ordered_dict")
    last_available_timestamp = journal_dict.peek_last_timestamp()
    new_journals = filtered_journal_df.published_at > last_available_timestamp
    if np.any(new_journals) > 0:
        # the journal_dict needs to be "topped up" with journals that were not included in filtered_journal_df when journal_dict was originally created
        # this occurs for example if loading from a checkpoint and the underlying journal data has expanded to include new dates or if the config was updated
        sdf = filtered_journal_df[new_journals].sort_values(by='published_at', ascending=True)
        assert len(sdf) > 0
        n_additions = 0
        for row in sdf.itertuples():
            payload = (row.user_id, row.site_id, row.journal_oid, row.published_at)
            journal_dict.insert_payload(row.published_at, payload)
            n_additions += 1
        logger.info(f"Added {n_additions} new journals from filtered_journal_df (n={len(filtered_journal_df)}) to journal_dict (n={len(journal_dict)}).")
    else:
        logger.info(f"Seemingly no new journals to add from filtered_journal_df (n={len(filtered_journal_df)}) to journal_dict (n={len(journal_dict)})")        


def get_usp_eligibility_dicts(user_site_df):
    """
    Deprecated.
    """
    logger = logging.getLogger("cbrec.timeAwareDict.get_usp_eligibility_dicts")
    # map of created_at -> (user_id, site_id)
    s = datetime.now()
    ts_to_first_update_dict = TimeAwareDict()
    ts_to_third_update_dict = TimeAwareDict()
    for row in user_site_df.sort_values(by='user_first_update_timestamp', ascending=True).itertuples():
        payload = (row.user_id, row.site_id)
        ts_to_first_update_dict.insert_payload(row.user_first_update_timestamp, payload)
    for row in user_site_df.sort_values(by='user_third_update_timestamp', ascending=True).itertuples():
        payload = (row.user_id, row.site_id)
        ts_to_third_update_dict.insert_payload(row.user_third_update_timestamp, payload)
    logger.info(f"Created user/site eligibility odicts with {len(ts_to_first_update_dict)} and {len(ts_to_third_update_dict)} distinct created_at values in {datetime.now() - s}.")
    return ts_to_first_update_dict, ts_to_third_update_dict
