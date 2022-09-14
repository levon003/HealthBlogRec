
from datetime import datetime
from collections import OrderedDict
import numpy as np

class RecentActivityManager:
    """
    Manages a collection of RecentActivityCounter objects.
    
    Maintains a list of active user ids across all activity types and provides a single method (`update_counts`) for updating all activity counters at once.

    TODO also track first journal times i.e. when a user starts existing. This is in addition to the info tracked in the journal_user RAC.
    We need this info for resolving some ties and for computing the ClosestToStart baseline
    """
    def __init__(self, config):
        if config is None:
            raise ValueError("Config object must be provided.")
        self.config = config
        activity_count_duration_ms = config.activity_count_duration_ms
        assert activity_count_duration_ms is not None
        interaction_count_duration_ms = config.interaction_count_duration_ms
        self.activity_counter_dict = {
            # these are for computing baselines
            'initiation_site': RecentActivityCounter(interaction_count_duration_ms),  # for the MostRecentInitiation and MostInitiations baselines
            'journal_site': RecentActivityCounter(interaction_count_duration_ms),  # for the MostRecentJournal and MostJournals baselines
            # these are the user activity counters
            'journal_user': RecentActivityCounter(activity_count_duration_ms),
            'amp': RecentActivityCounter(activity_count_duration_ms),
            'comment': RecentActivityCounter(activity_count_duration_ms),
            'guestbook': RecentActivityCounter(activity_count_duration_ms),
        }
        self.user_activity_keys = set(['journal_user', 'amp', 'comment', 'guestbook'])
        self.active_user_ids = set()

        self.first_journal_update_timestamp_dict = {}  # map of user_id -> timestamp
    
    def update_counts(self, current_timestamp):
        # update activity counters to the current moment
        all_removed_user_ids = set()
        for key, rac in self.activity_counter_dict.items():
            removed_user_ids = rac.update_counts(current_timestamp)
            if key in self.user_activity_keys:
                all_removed_user_ids |= removed_user_ids
        if len(all_removed_user_ids) > 0:
            # recompute active user ids
            # note: this might get a bit expensive
            self.active_user_ids = set()
            self.active_user_ids.update(*[
                self.activity_counter_dict[key].get_active_ids() 
                for key in self.user_activity_keys
            ])

            
    def get_active_user_ids(self):
        return self.active_user_ids
    
    def add_interaction(self, interaction_type: str, user_id: int, created_at: int):
        self.activity_counter_dict[interaction_type].add_interaction(user_id, created_at)
        self.active_user_ids.add(user_id)
        if interaction_type == 'journal_user':
            if user_id not in self.first_journal_update_timestamp_dict:
                self.first_journal_update_timestamp_dict[user_id] = created_at
                #print(f"First journal from {user_id} at {datetime.utcfromtimestamp(created_at / 1000)}.")

    def get_first_journal_update_timestamp(self, user_id):
        if user_id in self.first_journal_update_timestamp_dict:
            return self.first_journal_update_timestamp_dict[user_id]
        else:
            return None
        
    def get_activity_counter(self, interaction_type):
        return self.activity_counter_dict[interaction_type]
        
    def __repr__(self):
        summary = f"{len(self.activity_counter_dict)} activity counters ({len(self.user_activity_keys)} for users). Tracking {len(self.active_user_ids)} active users.\n"
        for int_type, rac in self.activity_counter_dict.items():
            summary += f"{int_type} recent activity: {len(rac.activity_count_dict)} unique users with {np.sum(list(rac.activity_count_dict.values()))} total interactions.\n"
        return summary
    
    def __str__(self):
        return self.__repr__()
        
            
class RecentActivityCounter():
    """
    Written for user_ids, but supports any form of hashable id, e.g. site_ids or (user_id, site_id) tuples
    """
    def __init__(self, activity_count_duration_ms):
        # map of created_at -> list(user_id)
        # tracks one type of activity
        # user_id is a list and not a set because a user may have multiple e.g. amps at the same timestamp
        self.ts_to_user_ids = OrderedDict()
        self.activity_count_dict = {}  # map of user_id -> int count
        self.activity_count_duration_ms = activity_count_duration_ms
        self.active_ids = set()  # tracks set of ids with non-zero counts in this RAC
        self.most_recent_activity_dict = OrderedDict()  # map of user_id -> created_at, sorted such that the last is the most recent
    
    def update_counts(self, current_timestamp):
        """
        Set the timestamp from which to give counts to current_timestamp,
        which has the effect of removing any old activity and updating the counts accordingly.
        Note: current_timestamp must be >= any previous calls to update_counts().
        
        Returns user_ids no longer considered active.
        """
        expired_timestamp = current_timestamp - self.activity_count_duration_ms
        removed_ids = set()
        while len(self.ts_to_user_ids) > 0:
            if next(iter(self.ts_to_user_ids)) < expired_timestamp:
                # this activity has expired
                _, user_id_list = self.ts_to_user_ids.popitem(last=False)
                # update the counts to account for the removal
                for user_id in user_id_list:
                    self.activity_count_dict[user_id] -= 1
                    # delete old keys when count hits 0
                    if self.activity_count_dict[user_id] == 0:
                        del self.activity_count_dict[user_id]
                        del self.most_recent_activity_dict[user_id]
                        self.active_ids.remove(user_id)
                        removed_ids.add(user_id)
            else:
                break
        return removed_ids
    
    def add_interaction(self, user_id, created_at):
        """
        Add an interaction from user_id at time created_at to the activity tracker.
        
        """
        if created_at in self.ts_to_user_ids:
            user_id_list = self.ts_to_user_ids[created_at]
        else:
            user_id_list = []
            self.ts_to_user_ids[created_at] = user_id_list
        user_id_list.append(user_id)
        if user_id in self.activity_count_dict:
            self.activity_count_dict[user_id] += 1
        else:
            self.activity_count_dict[user_id] = 1
            self.active_ids.add(user_id)
        # keep track of users by the recency of their activity
        self.most_recent_activity_dict[user_id] = created_at
        self.most_recent_activity_dict.move_to_end(user_id)
    
    def get_count(self, user_id):
        """
        Activity count for a user_id in the last self.activity_count_duration_ms milliseconds
        """
        if user_id not in self.activity_count_dict:
            return 0
        return self.activity_count_dict[user_id]
    
    def get_most_recent_activity(self, user_id):
        """
        created_at timestamp of most recent activity for a user_id with a non-zero count.
        Returns None if no recent activity for user_id.
        """
        if user_id in self.most_recent_activity_dict:
            return self.most_recent_activity_dict[user_id]
        else:
            return None
        
    def get_active_ids_by_recency(self, include_timestamps=True):
        """
        Sorts all active user ids (those in self.most_recent_activity_dict) by their value in most_recent_activity_dict.
        
        Useful for computing a MostRecent baseline.
        
        :include_timestamps: If True, returns (user_id, timestamp) tuples, else returns just a list of user_ids.
        """
        active_users = [user_id for user_id in reversed(self.most_recent_activity_dict)]
        if include_timestamps:
            return [(user_id, self.most_recent_activity_dict[user_id]) for user_id in active_users]
        else:
            return active_users
                
    def get_active_ids_by_count(self, include_counts=True):
        """
        Sorts all active user ids (those in self.activity_count_dict) by their value in activity_count_dict.
        
        Useful for computing a MostActive baseline.
        
        :include_counts: If True, returns (user_id, count) tuples, else returns just a list of user_ids.
        """
        active_users = [(user_id, count) for user_id, count in self.activity_count_dict.items()]
        active_users.sort(key=lambda tup: tup[1], reverse=True)
        if include_counts:
            return active_users
        else:
            return [user_id for user_id, _ in active_users]
    
    def get_active_ids(self):
        return self.active_ids
    
    def is_active(self, user_id):
        return user_id in self.active_ids