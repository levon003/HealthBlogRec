from . import featuredb

import logging
import numpy as np

class RecContext:
    """
    Captures the context of a recommendation associated with a particular initiation.

    Could be used for arbitrary recommendation, but currently the notion of a target/"correct" site is baked in. Perhaps one could just choose a random site, but in general this suggests re-architecting.

    Idea: three "modes": train, test, predict
    """
    def create_from_test_context(config, md: dict, test_context):
        """
        Create and return a RecContext instance from a metadata dictionary and a test_context stored in the database.
        """
        assert test_context['metadata_id'] == md['metadata_id']
        metadata_id = test_context['metadata_id']
        source_usp_arr = test_context['source_usp_arr'].astype(np.int64)  # array of (user_id, site_id) pairs, with shape (X, 2), most commonly (1, 2)
        candidate_usp_arr = test_context['candidate_usp_arr'].astype(np.int64)  # with shape (Y, 2)
        target_inds = test_context['target_inds'].astype(np.int64)  # array of integer indices, corresponding to candidate_usp_arr, in shape (Z,)

        source_usp_mat = test_context['source_usp_mat']  # source_usp_arr.shape[0] x (9+3), most commonly (1, 12)
        candidate_usp_mat = test_context['candidate_usp_mat']  # (Y, 12)
        user_pair_mat = test_context['user_pair_mat']  # (X * Y, 3)

        timestamp = md['timestamp']
        source_user_id = md['source_user_id']
        target_site_id = md['target_site_id'] if md['has_target'] else None
        rc = RecContext(config, metadata_id, timestamp, source_user_id, target_site_id)
        rc.md.update(md)
        rc.is_initiation_eligible = md['is_initiation_eligible']

        rc.source_usp_arr = source_usp_arr
        rc.candidate_usp_arr = candidate_usp_arr
        rc.target_inds = target_inds
        rc.source_usp_mat = source_usp_mat
        rc.candidate_usp_mat = candidate_usp_mat
        rc.user_pair_mat = user_pair_mat
        return rc

    def __init__(self, config, metadata_id, timestamp, source_user_id: int, target_site_id):
        self.config = config
        self.metadata_id = metadata_id
        self.timestamp = timestamp
        self.source_user_id = source_user_id
        self.target_site_id = target_site_id

        self.alt_usps = None  # only defined if train features are generated

        # metadata dictionary; map of string keys to primitive values
        self.md = {
            'metadata_id': metadata_id,
            'timestamp': timestamp,
            'source_user_id': int(source_user_id),
        }

        self.has_target = target_site_id is not None
        self.md['has_target'] = self.has_target
        if self.has_target:
            self.md['target_site_id'] = target_site_id

        # determine if this is during test time
        self.is_test_period = self.timestamp >= self.config.test_generation_start_timestamp
        self.md['is_test_period'] = self.is_test_period

    def identify_eligibility(self, eligibility_manager):
        """
        Sets self.is_initiation_eligible
        """
        self.source_site_ids = eligibility_manager.get_eligible_sites_from_user(self.source_user_id)
        self.md['n_source_sites'] = len(self.source_site_ids)
        if self.has_target:
            self.target_user_ids = eligibility_manager.get_eligible_users_on_site(self.target_site_id)
            self.md['n_target_users'] = len(self.target_user_ids)

        # if the user isn't eligible, determine if the source user exists as an author (yet?)
        self.md['source_user_is_existing'] = len(self.source_site_ids) > 0 or eligibility_manager.is_user_existing(self.source_user_id)
        # determine how many existing users there are on the target site.
        # note: if n_target_users < n_existing_users_on_target_site, then:
        #            if n_target_users >= 1: mix of eligible and existing users on this site
        #            if n_target_users == 0: no one on the site is eligible yet
        if self.has_target:
            self.md['n_existing_users_on_target_site'] = len(eligibility_manager.get_existing_users_on_site(self.target_site_id))

        user_is_eligible = len(self.source_site_ids) > 0
        self.md['source_user_is_eligible'] = user_is_eligible
        if self.has_target:
            site_has_eligible_user = len(self.target_user_ids) > 0
            is_self_initiation = bool(np.any([user_id == self.source_user_id for user_id in self.target_user_ids]))
            self.md['target_site_has_eligible_user'] = site_has_eligible_user
            self.md['is_self_initiation'] = is_self_initiation
        else:  # no target site
            site_has_eligible_user = False
            is_self_initiation = False

        # determine if initiation is eligible for recommendation
        # in other words, can the model actually generate scores for this recommendation and should it be taken as positive implicit feedback?
        if self.has_target:
            self.is_initiation_eligible = user_is_eligible and site_has_eligible_user and not is_self_initiation
        else:  # no target
            self.is_initiation_eligible = user_is_eligible
        self.md['is_initiation_eligible'] = self.is_initiation_eligible

        # identify other eligible users
        # but only if the initiation is eligible for recommendation
        if self.is_initiation_eligible:
            self.eligible_user_ids = eligibility_manager.get_eligible_user_ids().copy()

            # remove the source user
            #assert self.source_user_id in self.eligible_user_ids, "Source user must be eligible."
            self.eligible_user_ids.remove(self.source_user_id)

            # remove any target users
            if self.has_target:
                for user_id in self.target_user_ids:
                    #assert user_id in self.eligible_user_ids, "Target users must be eligible."
                    self.eligible_user_ids.remove(user_id)

            # remove any co-authors
            n_eligible_coauthors = 0
            for user_id in eligibility_manager.get_eligible_coauthors(self.source_user_id):
                if user_id in self.eligible_user_ids:  # while all coauthors are eligible, they might already have been removed as target_user_ids
                    self.eligible_user_ids.remove(user_id)
                    n_eligible_coauthors += 1
            self.md['n_eligible_coauthors'] = n_eligible_coauthors

            self.md['n_eligible_users'] = len(self.eligible_user_ids)
        
    def identify_usps(self, activity_manager, graph, eligibility_manager):
        """
        For an self.is_initiation_eligible rec context, identifies USPs, including alts/candidates.

        Sensitive to train/test split. 
        :returns -- True if eligible, False if not
        """

        if not self.is_initiation_eligible:
            return False

        # identify usp sources and targets
        # sources is every usp that has this user_id
        self.source_usps = [(self.source_user_id, site_id) for site_id in self.source_site_ids]
        self.md['n_source_usps'] = len(self.source_usps)  # at original writing, redundant with n_source_sites, but this could change in the future

        # bring in activity info
        # to select alternatives, we need to identify ACTIVE eligible users
        # active means "amp, comment, guestbook, or journal" in last X milliseconds
        active_user_ids = activity_manager.get_active_user_ids()
        self.md['n_active_user_ids'] = len(active_user_ids)

        # identify whether the source and targets were active
        self.md['source_user_is_active'] = self.source_user_id in active_user_ids
        if self.has_target:
            self.md['n_active_target_users'] = sum([1 if user_id in active_user_ids else 0 for user_id in self.target_user_ids])
            # note: if n_active_target_users == 0, then this site would not have been eligible to be recced (and thus from an eval perspective we fail). 
            # If n_active_target_users < n_target_users, this is also bad: it means that we are "cheating" if we rank and combine all USPs on self.target_site_id, as in reality we would NOT have scored all USPs on this site
            # Thus, we should remove non-active USPs that include the target site
            # This is done in testing but not in training.
            if self.is_test_period:
                # remove non-active USPs that include the target site
                # HOWEVER, if there are no active USPs that include the target site, we force one into the list for evaluation reasons
                # in most cases, this selects the only eligible user on the site. 
                # However, in the case of multiple eligible users, it selects the one who joined the site most recently (which is a very arbitrary selection criterion; random might literally be better. Q: how often does this happen?)
                self.target_usps = [(user_id, self.target_site_id) for user_id in self.target_user_ids if user_id in active_user_ids]
                if len(self.target_usps) == 0:
                    if len(self.target_user_ids) == 1:
                        # add it anyway, even though the user isn't active
                        self.target_usps = [(list(self.target_user_ids)[0], self.target_site_id),]
                        test_target_usp_adjustment = 'forced_only_user'
                    else:
                        # select the one that joined the site most recently
                        target_user_id_list = [user_id for user_id in self.target_user_ids]
                        i = np.argmax([activity_manager.get_first_journal_update_timestamp(user_id) for user_id in target_user_id_list])
                        self.target_usps = [(target_user_id_list[i], self.target_site_id),]
                        test_target_usp_adjustment = 'forced_most_recent_user'
                else:  # at least one user active on the site
                    test_target_usp_adjustment = 'none'
                self.md['test_target_usp_adjustment'] = test_target_usp_adjustment

                self.md['source_user_initiated_in_train_period'] = eligibility_manager.did_user_initiate_in_training_period(self.source_user_id)
                self.md['target_site_initiated_with_in_train_period'] = eligibility_manager.was_site_initiated_with_in_training_period(self.target_site_id)
            else:
                # this is training, so include all target_usps
                # TODO it's possible that we should only include 1 or we should cap the number of target_usps generated in this way
                self.target_usps = [(user_id, self.target_site_id) for user_id in self.target_user_ids]
            self.md['n_target_usps'] = len(self.target_usps)


        eligible_but_inactive = self.eligible_user_ids - active_user_ids
        self.md['n_eligible_inactive_users'] = len(eligible_but_inactive)

        candidate_user_ids = active_user_ids & self.eligible_user_ids

        # finally, remove from active any user_ids invalid for this source user_id
        # invalid users include:
        #  - the source user_id
        #  - any user_id who has published on the target site_id
        #  - any user_id that int_user_id has previously connected with
        # We only need to remove the 3rd at this stage, as self.eligible_user_ids is set to remove the first two in `identify_usps`
        existing_initiations_from_source_user_id = graph.get_edge_targets_for_user_id(self.source_user_id)
        self.md['n_existing_initiations_from_source_user_id'] = len(existing_initiations_from_source_user_id)
        if len(existing_initiations_from_source_user_id) > 0:
            candidate_user_ids -= existing_initiations_from_source_user_id
        self.md['n_candidate_user_ids'] = len(candidate_user_ids)

        # generate the usps from these active users
        # we use an array to make sampling faster
        self.candidate_usps = [
            (candidate_user_id, site_id)
            for candidate_user_id in candidate_user_ids
            for site_id in eligibility_manager.get_eligible_sites_from_user(candidate_user_id)
        ]
        if len(self.candidate_usps) == 0:
            raise ValueError("No active usps! That means no alternatives, which should never happen.")
        self.md['n_candidate_usps'] = len(self.candidate_usps)

        return True


    def generate_features(self, feature_generator, use_cache=None):
        """
        Generates and saves in the database the features for this rec context; depends on train/test split.
        """
        if not self.is_initiation_eligible:
            raise ValueError("Bad RecContext; expected eligibility.")
        if not self.has_target and not self.is_test_period:
            logger = logging.getLogger('cbrec.reccontext.generate_features')
            logger.warning("Forcing target-less reccontext to be in the test period; potential problem.")
            self.is_test_period = True

        if self.is_test_period:
            self.generate_test_features(feature_generator, use_cache=use_cache)
        else:  # is training
            self.generate_train_features(feature_generator)


    def generate_test_features(self, feature_generator, use_cache=None):
        self.source_usp_arr = np.array(self.source_usps)
        if self.has_target:
            self.candidate_usp_arr = np.array(self.candidate_usps + self.target_usps)  # candidate array contains ALL candidates, including any target candidates
            # we keep track of the indices that refer to any targets
            self.target_inds = np.arange(len(self.candidate_usps), len(self.candidate_usp_arr))
            use_cache = False if use_cache is None else use_cache  # no probable speedup to cache use, so save time and memory and force a recompute
        else:  # no target
            self.candidate_usp_arr = np.array(self.candidate_usps)
            self.target_inds = np.array([])
            use_cache = True if use_cache is None else use_cache  # probable cache speedup while doing predictions

        # now sort so that every USP for a site is clumped together
        sort_inds = self.candidate_usp_arr[:,1].argsort()
        self.candidate_usp_arr = self.candidate_usp_arr[sort_inds]
        # update which inds contain the target (if any)
        n_target_inds = len(self.target_inds)
        self.target_inds = np.argwhere(np.isin(sort_inds, self.target_inds)).ravel()
        assert len(self.target_inds) == n_target_inds

        source_usp_mat = np.empty((len(self.source_usp_arr), self.config.user_feature_count), dtype=featuredb.NUMPY_DTYPE)
        for i, source_usp in enumerate(self.source_usp_arr):
            source_usp_arr = feature_generator.generate_user_features(source_usp, self.timestamp, use_cache=use_cache)
            source_usp_mat[i,:] = source_usp_arr
        self.source_usp_mat = source_usp_mat

        candidate_usp_mat = np.empty((len(self.candidate_usp_arr), self.config.user_feature_count), dtype=featuredb.NUMPY_DTYPE)
        for i, candidate_usp in enumerate(self.candidate_usp_arr):
            candidate_usp_arr = feature_generator.generate_user_features(candidate_usp, self.timestamp, use_cache=use_cache)
            candidate_usp_mat[i,:] = candidate_usp_arr
        self.candidate_usp_mat = candidate_usp_mat

        user_pair_mat = np.empty((len(self.source_usp_arr) * len(self.candidate_usp_arr), self.config.user_pair_feature_count), dtype=featuredb.NUMPY_DTYPE)
        for i, source_usp in enumerate(self.source_usp_arr):
            for j, candidate_usp in enumerate(self.candidate_usp_arr):
                feature_pair_arr = feature_generator.generate_user_pair_features(source_usp, candidate_usp, self.timestamp)
                ind = (i * len(self.candidate_usp_arr)) + j
                user_pair_mat[ind,:] = feature_pair_arr
        self.user_pair_mat = user_pair_mat

    def generate_train_features(self, feature_generator):
        if not self.has_target:
            raise ValueError("Tried to generate train features for a targetless RecContext.")
        # create all combinations of source and target usps
        self.triples = []

        # avoid duplicate work by generating each target USP features only once
        target_feat_arr_dict = {}
        for target_usp in self.target_usps:
            target_feat_arr = feature_generator.generate_user_features(target_usp, self.timestamp)
            target_feat_arr_dict[target_usp] = target_feat_arr

        self.alt_usps = []
        for source_usp in self.source_usps:
            source_feat_arr = feature_generator.generate_user_features(source_usp, self.timestamp)
            for target_usp in self.target_usps:
                target_feat_arr = target_feat_arr_dict[target_usp]

                # select an alternative (alt) usp
                alt_usp = self.config.rng.choice(self.candidate_usps)
                alt_feat_arr = feature_generator.generate_user_features(alt_usp, self.timestamp)
                self.alt_usps.append(alt_usp)

                source_target_feat_arr = feature_generator.generate_user_pair_features(source_usp, target_usp, self.timestamp)
                source_alt_feat_arr = feature_generator.generate_user_pair_features(source_usp, alt_usp, self.timestamp)

                self.triples.append((
                    source_usp, target_usp, alt_usp, source_feat_arr, target_feat_arr, alt_feat_arr, source_target_feat_arr, source_alt_feat_arr
                ))
    
    def record_journal_ids(self, eligibility_manager):
        """
        Record the journal OIDs for the source, target, and alt USPs to the metadata.
        
        """
        # save the journal_oids for most recent journals made by the source, target, and candidates
        usp_names = ['source_usp',]
        usp_lists = [self.source_usps, ]
        if self.has_target:  # only save target USP journal_oids if there is a target
            usp_names.append('target_usp')
            usp_lists.append(self.target_usps)
        if self.alt_usps is not None:
            usp_names.append('alt_usp')
            usp_lists.append([tuple(usp) for usp in self.alt_usps])
        # TODO need to record journal ids for self.candidate_usps as well, in order to do proper testing
        # alternative: testing candidates are just any active user, so it's easy to identify the journals
        # that are needed during testing: it's recent updates from any active user!  
        for usp_name, usp_list in zip(usp_names, usp_lists):
            usp_recent_journals_map = {}
            self.md[usp_name + '_recent_journals'] = usp_recent_journals_map
            for usp in usp_list:
                journal_oids = eligibility_manager.get_recent_journal_update_oids(usp)
                usp_recent_journals_map[str(usp)] = journal_oids