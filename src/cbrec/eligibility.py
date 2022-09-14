from collections import defaultdict
import logging

class UserSitePairEligibilityManager:
    """
    The USP Eligibility Manager tracks existing and eligible users.
    
    Confusingly, it also tracks existing interactions in terms of which sites have been interacted with my which users (which is needed for graph-generation reasons).
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('cbrec.eligibility.UserSitePairEligibilityManager')
        
        self.eligible_usps = set()
        self.eligible_user_ids = set()
        self.existing_user_site_map = defaultdict(set)  # map of user_id -> set(site_id), where user_id is an existing author on the set of sites
        self.eligible_user_site_map = defaultdict(set)  # map of user_id -> set(site_id), where user_id is an eligible author on the set of sites
        self.existing_site_user_map = defaultdict(set)  # map of site_id -> set(user_id), where site_id has the set of existing authors
        self.eligible_site_user_map = defaultdict(set)  # map of site_id -> set(user_id), where site_id has the set of eligible authors
        
        self.site_user_int_dict = defaultdict(set)  # map of site_id -> set(user_id), where each user_id has interacted with this site_id

        # map of (user_id, site_id) -> [journal_oid, ...] of most recent updates, from most to least recent, from 1 to config.journal_update_memory stored
        self.usp_most_recent_update = {}
        self.journal_update_memory = self.config.journal_update_memory
        
        self.usp_count_dict = {}  # map of (user_id, site_id) -> count of journal updates

        self.initiating_user_ids = set()  # set of user_ids that have initiated

        # track which ids were present in the training period
        self.train_initiated_with_site_ids = set()
        self.train_initiating_user_ids = set()
    
    def get_eligible_user_ids(self):
        """
        Note: returns a direct reference to the underlying set. Make a copy before manipulating the set with `add` or `remove`.
        """
        return self.eligible_user_ids
        
    def add_existing_usp(self, user_id, site_id):
        self.existing_user_site_map[user_id].add(site_id)
        self.existing_site_user_map[site_id].add(user_id)
        
    def add_eligible_usp(self, user_id, site_id):
        self.eligible_usps.add((user_id, site_id))
        self.eligible_user_ids.add(user_id)
        self.eligible_user_site_map[user_id].add(site_id)
        self.eligible_site_user_map[site_id].add(user_id)
        
    def get_existing_users_on_site(self, site_id):
        if site_id in self.existing_site_user_map:
            curr_existing_users_on_site = self.existing_site_user_map[site_id]
        else:
            curr_existing_users_on_site = []
        return curr_existing_users_on_site
    
    def get_eligible_users_on_site(self, site_id):
        if site_id in self.eligible_site_user_map:
            curr_eligible_users_on_site = self.eligible_site_user_map[site_id]
        else:
            curr_eligible_users_on_site = []
        return curr_eligible_users_on_site
    
    def get_existing_sites_from_user(self, user_id):
        """
        Returns sites on which this user has authored updates.
        Note: can include sites on which the user will never author 3+ updates
        """
        if user_id in self.existing_user_site_map:
            curr_existing_sites_from_user = self.existing_user_site_map[user_id]
        else:
            curr_existing_sites_from_user = []
        return curr_existing_sites_from_user

    def is_user_existing(self, user_id):
        return user_id in self.existing_user_site_map
    
    def get_eligible_sites_from_user(self, user_id):
        """
        Returns sites on which this user has authored 3+ updates.
        """
        if user_id in self.eligible_user_site_map:
            curr_eligible_sites_from_user = self.eligible_user_site_map[user_id]
        else:
            curr_eligible_sites_from_user = []
        return curr_eligible_sites_from_user

    def is_user_eligible(self, user_id):
        return user_id in self.eligible_user_site_map

    def get_recent_journal_update_oids(self, usp):
        if usp in self.usp_most_recent_update:
            return self.usp_most_recent_update[usp]
        else:
            raise ValueError(f"{usp} not among the {len(self.usp_most_recent_update)} user/site pairs with recent updates.")
        
    def add_recent_journal_update(self, user_id, site_id, journal_oid):
        usp = (user_id, site_id)
        if usp in self.usp_most_recent_update:
            self.usp_most_recent_update[usp] = [journal_oid,] + self.usp_most_recent_update[usp][:self.journal_update_memory-1]
            # For generating list for extraction of participants, should run a bespoke process:
                # identify all eligible/active authors AND participants
                # pull three most recent at end-of-snapshot timestamp
                # do something using nlargest (if it's performant enough)?
        else:
            self.usp_most_recent_update[usp] = [journal_oid,]
            
        # update usp counts and existing/eligibility status
        is_new_existing = False
        is_new_eligible = False
        if usp not in self.eligible_usps:
            if usp in self.usp_count_dict:
                self.usp_count_dict[usp] += 1
                if self.usp_count_dict[usp] == 3:
                    # this user has become eligible!
                    is_new_eligible = True
                    self.add_eligible_usp(user_id, site_id)
                    # once a USP becomes eligible, we stop tracking its count
                    # assert usp in self.eligible_usps
                    del self.usp_count_dict[usp]
            else:
                self.usp_count_dict[usp] = 1
                self.add_existing_usp(user_id, site_id)
                is_new_existing = True
        return is_new_existing, is_new_eligible
        
    def get_existing_site_ints(self, site_id):
        prev_user_ids = self.site_user_int_dict[site_id]
        return prev_user_ids
    
    def add_site_int(self, user_id, site_id):
        self.site_user_int_dict[site_id].add(user_id)
        self.initiating_user_ids.add(user_id)

    def is_initiation(self, user_id, site_id):
        """
        If this is the first time this user_id has interacted with this site_id
        """
        return user_id not in self.site_user_int_dict[site_id]

    def has_user_initiated(self, user_id):
        return user_id in self.initiating_user_ids

    def get_eligible_coauthors(self, target_user_id):
        """
        Identify all ELIGIBLE user ids who have authored on EXISTING sites authored on by target_user_id.

        :returns -- a set of user_ids
        """
        coauthor_ids = set()
        for site_id in self.get_existing_sites_from_user(target_user_id):
            for user_id in self.get_eligible_users_on_site(site_id):
                if user_id == target_user_id:
                    continue
                coauthor_ids.add(site_id)
        return coauthor_ids

    def maintain_user_count(self, user_id, is_source, is_target, is_active):
        """
        Within the test data, we want to store user exposure counts to compute coverage.
        Specifically, we need:
        -Number of times a site has no (active/eligible) users when an initiation occurs
        -Number of times a source is not active when they initiate (but I think this is tracked elsewhere...)
        -Number of times a source is not (existing/eligible) when they inititate (is tracked elsewhere, i.e. can be computed from the initiation metadata)
        -Number of 
        """
        if is_source or is_target:
            is_existing = True
            is_eligible = True
        else:
            is_existing = user_id in self.existing_user_site_map
            is_eligible = is_existing and user_id in self.eligible_user_site_map
        # TODO update the counters based on existing/eligible/active/source/target status...

    def update_training_period_ids(self):
        """
        Should be called once when the training period is finished and the testing period is about to begin.

        We could consider many things:
        - USERS initiating in the training period
        - SITES initiated with in the training period
        And the extrapolation to USPs:
        - SITES with initiating authors in the training period
        - USERS on sites that were initiated with in the training period

        For now, we only track the first two points.
        """
        self.train_initiated_with_site_ids = set(self.site_user_int_dict.keys())
        self.train_initiating_user_ids = self.initiating_user_ids.copy()
        self.logger.info(f"Identified {len(self.train_initiating_user_ids)} initiating users and {len(self.train_initiated_with_site_ids)} initiated-with sites in the training period.")

    def did_user_initiate_in_training_period(self, user_id):
        return user_id in self.train_initiating_user_ids

    def was_site_initiated_with_in_training_period(self, site_id):
        return site_id in self.train_initiated_with_site_ids
