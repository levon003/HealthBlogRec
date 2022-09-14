"""
The goal of this module is to provide a loader that manages and combines feature data from various sources, including:
- Train triples
- Test contexts with targets
- Test contexts without targets (for prediction)


"""

import os
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime

from . import featuredb
from . import reccontext
from . import evaluation
from . import data
from .text import embeddingdb
from .text import journalid


class FeatureLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("cbrec.feature_loader.FeatureLoader")
        
        
        dm = data.DataManager(config, load_ints=False, load_journals=True)
        self.journal_id_lookup = journalid.JournalIdLookup(config, dm)
        
        # rec_input_matrix_cache is a map of metadata_id -> feature matrix
        # TODO should use a real cache, and probably add a flag to turn off prediction feature caching by default
        self.rec_input_matrix_cache = {}
                
    
    def get_pointwise_training_triples(self):
        db = featuredb.get_db_by_filepath(self.config.feature_db_filepath)
        with db:
            feature_arrs, ys, missing_journal_id_list = self.get_input_arrs_from_triple_dicts(featuredb.stream_triples(db))
        y_true = np.array(ys)
        if len(feature_arrs) > 0:
            X = np.vstack(feature_arrs)
        else:
            X = np.array(feature_arrs)
        return X, y_true, missing_journal_id_list
    
    
    def identify_required_journal_ids(self, md_list):
        required_journal_ids = set()
        db = featuredb.get_db_by_filepath(self.config.feature_db_filepath)
        n_invalid = 0
        with db:
            for md in tqdm(md_list, desc="Identifying required journal ids"):
                metadata_id = md['metadata_id']
                test_context = featuredb.get_test_context_by_metadata_id(db, metadata_id, self.config)
                interaction_timestamp = int(md['timestamp'])
                
                source_usp_arr = test_context['source_usp_arr'].astype(np.int64)
                source_usps = [(source_usp_arr[i,0], source_usp_arr[i,1]) for i in range(source_usp_arr.shape[0])]
                
                candidate_usp_arr = test_context['candidate_usp_arr'].astype(np.int64)
                candidate_usps = [(candidate_usp_arr[i,0], candidate_usp_arr[i,1]) for i in range(candidate_usp_arr.shape[0])]
                
                for usp in source_usps + candidate_usps:
                    journal_ids = self.journal_id_lookup.get_journal_updates_before(usp, interaction_timestamp)
                    if len(journal_ids) < 3:
                        n_invalid += 1
                    else:
                        required_journal_ids.update(journal_ids)
                        
        self.logger.info(f"identify_required_journal_ids - Identified {len(required_journal_ids)} required journal ids ({n_invalid} invalid USPs)")
        return required_journal_ids
        
    
    def get_reccontexts_from_test_contexts(self, test_md_list, site_allowlist=None, verify_text_available=True, omit_invalid_texts=True):
        """
        
        if omit_invalid_texts is False, an error will be thrown if the correct texts are not available.
        
        :returns - list of  RecContexts vectors, where RecContext.X_test is the feature matrix as returned from self.get_input_matrix_from_test_context
        """
        # define a helper function for filtering via the allowlist
        def is_usp_allowed(usp):
            if site_allowlist is None:
                return True
            else:
                return int(usp[1]) in site_allowlist
        self.logger.info(f"Generating RecContexts with features for {len(test_md_list)} test contexts.")
        rc_list = []
        db = featuredb.get_db_by_filepath(self.config.feature_db_filepath)
        with db:
            for md in tqdm(test_md_list, desc="Creating test RecContexts"):
                metadata_id = md['metadata_id']
                test_context = featuredb.get_test_context_by_metadata_id(db, metadata_id, self.config)
                # Verify and remove candidate USPs that don't have available texts
                # should probably keep the site_allowlist though
                candidate_usp_arr = test_context['candidate_usp_arr'].astype(np.int64)
                candidate_usps = [(candidate_usp_arr[i,0], candidate_usp_arr[i,1]) for i in range(candidate_usp_arr.shape[0])]
                invalid_mask = np.array([not is_usp_allowed(usp) for usp in candidate_usps])
                if np.sum(invalid_mask) > 0:
                    self.logger.info(f"Removing {np.sum(invalid_mask)} / {len(candidate_usp_arr)} candidate USPs that are not on the allowlist.")
                    test_context['candidate_usp_arr'] = candidate_usp_arr[~invalid_mask]
                    test_context['candidate_usp_mat'] = test_context['candidate_usp_mat'][~invalid_mask]
                    
                if verify_text_available:
                    candidate_usps = [(usp[0], usp[1]) for usp in test_context['candidate_usp_arr'].astype(np.int64)]
                    n_invalid = np.sum([len(self.journal_id_lookup.get_journal_updates_before(usp, md['timestamp'])) < 3 for usp in candidate_usps])
                    if n_invalid > 0:
                        if omit_invalid_texts:
                            candidate_usp_arr = test_context['candidate_usp_arr'].astype(np.int64)
                            candidate_usps = [(candidate_usp_arr[i,0], candidate_usp_arr[i,1]) for i in range(candidate_usp_arr.shape[0])]
                            invalid_mask = np.array([len(self.journal_id_lookup.get_journal_updates_before(usp, md['timestamp'])) < 3 for usp in candidate_usps])
                            if np.sum(invalid_mask) == 0:
                                self.logger.debug("Warning: Unexpected lack of invalid entries in mask.")
                            self.logger.info(f"Removing {np.sum(invalid_mask)} / {len(candidate_usp_arr)} candidate USPs that don't have available texts.")
                            test_context['candidate_usp_arr'] = candidate_usp_arr[~invalid_mask]
                            test_context['candidate_usp_mat'] = test_context['candidate_usp_mat'][~invalid_mask]
                        else:
                            raise ValueError(f"Identified {n_invalid} candidate USPs without sufficient texts available.")
                                   
                rc = reccontext.RecContext.create_from_test_context(self.config, md, test_context)
                try:
                    # add the X_test key to the RecContext, which the Torch scoring model uses for prediction
                    rc.X_test = self.get_input_matrix_from_test_context(rc)  # force cache generation of the features for this rec context
                    rc_list.append(rc)
                except Exception as ex:
                    self.logger.error(ex)
                    raise ex
                    continue
        return rc_list
    
    
    def create_train_triples_from_test_contexts(self, test_md_list):
        """
        This function does two things:
         - Creates training triples by negative sampling from test context candidate USPs
         - Identifies journal updates needed to generate features for those training triples
        
        It returns the training triples such that they can be passed to self.get_input_arrs_from_triple_dicts([list of d]),
        where d must contain:
         - {source,target,alt}_user_id
         - {source,target,alt}_site_id
         - interaction_timestamp
         - {source,target,alt}_feature_arr
         - source_target_feature_arr
         - source_alt_feature_arr
        """
        self.logger.info(f"Creating training triples from {len(test_md_list)} test contexts.")
        triple_dicts = []
        required_journal_ids = set()  # set of the journal ids required to generate features for the created triples
        rng = np.random.default_rng(12)  # use a new default_rng instance to ensure the same alternatives will be selected when given the same set of test contexts
        db = featuredb.get_db_by_filepath(self.config.feature_db_filepath)
        with db:
            for md in tqdm(test_md_list, desc="Creating training triples"):
                metadata_id = md['metadata_id']
                try:
                    test_context = featuredb.get_test_context_by_metadata_id(db, metadata_id, self.config)
                except ValueError as ex:
                    self.logger.warning(f"Failed to retrieve test context with metadata_id={metadata_id}.")
                    continue
                interaction_timestamp = int(md['timestamp'])
                
                source_usp_arr = test_context['source_usp_arr'].astype(np.int64)
                candidate_usp_arr = test_context['candidate_usp_arr'].astype(np.int64)
                
                source_usp_mat = test_context['candidate_usp_mat']
                candidate_usp_mat = test_context['candidate_usp_mat']
                user_pair_mat = test_context['user_pair_mat']
                
                target_inds = test_context['target_inds'].astype(np.int64)
                
                # Identify indices that are appropriate to use as alternatives
                # (which is any non-target)
                alt_candidate_mask = np.ones(len(candidate_usp_arr)).astype(bool)
                alt_candidate_mask[target_inds] = False
                alt_candidate_inds = np.arange(len(candidate_usp_arr))[alt_candidate_mask]
                
                for i, source_usp in enumerate(source_usp_arr):
                    source_journal_ids = self.journal_id_lookup.get_journal_updates_before(tuple(source_usp), interaction_timestamp)
                    if len(source_journal_ids) < 3:
                        self.logger.warning(f"Source USP {source_usp} has <3 journals available, despite being generated as test context {metadata_id}.")
                        continue
                    
                    # extract the source features for this USP
                    source_feature_arr = source_usp_mat[i,:]
                    
                    for target_ind in target_inds:
                        target_usp = candidate_usp_arr[target_ind]
                        
                        # Extract target arrs from among the candidates
                        target_feature_arr = candidate_usp_mat[target_ind,:]
                        user_pair_ind = (i * len(candidate_usp_arr)) + target_ind
                        source_target_feature_arr = user_pair_mat[user_pair_ind,:]

                        # Sample an alt from among the candidates
                        alt_ind = rng.choice(alt_candidate_inds)
                        alt_usp = candidate_usp_arr[alt_ind]
                        
                        # Extract alt arrs from among the candidates
                        alt_feature_arr = candidate_usp_mat[alt_ind,:]
                        user_pair_ind = (i * len(candidate_usp_arr)) + alt_ind
                        source_alt_feature_arr = user_pair_mat[user_pair_ind,:]
                        
                        # Identify target and alt required journal ids
                        target_journal_ids = self.journal_id_lookup.get_journal_updates_before(tuple(target_usp), interaction_timestamp)
                        if len(target_journal_ids) < 3:
                            self.logger.warning(f"Target USP {target_usp} has <3 journals available, despite being generated as a candidate in context {metadata_id}.")
                            continue
                        alt_journal_ids = self.journal_id_lookup.get_journal_updates_before(tuple(alt_usp), interaction_timestamp)
                        if len(alt_journal_ids) < 3:
                            self.logger.warning(f"Alt USP {alt_usp} has <3 journals available, despite being generated as a candidate in context {metadata_id}.")
                            continue
                        
                        d = {
                            'source_user_id': source_usp[0],
                            'source_site_id': source_usp[1],
                            'target_user_id': target_usp[0],
                            'target_site_id': target_usp[1],
                            'alt_user_id': alt_usp[0],
                            'alt_site_id': alt_usp[1],
                            'interaction_timestamp': interaction_timestamp,
                            'source_feature_arr': source_feature_arr,
                            'target_feature_arr': target_feature_arr,
                            'alt_feature_arr': alt_feature_arr,
                            'source_target_feature_arr': source_target_feature_arr,
                            'source_alt_feature_arr': source_alt_feature_arr,
                        }
                        triple_dicts.append(d)
                        required_journal_ids.update(source_journal_ids, target_journal_ids, alt_journal_ids)
        return triple_dicts, required_journal_ids
    
    
    def combine_feature_arrs(self, source_feature_arr, candidate_feature_arr, source_candidate_feature_arr, source_text_arrs, candidate_text_arrs):
        """
        This function creates a fixed-length input to a model, given non-text feature arrs and text embeddings.
        
        If either of the text embeddings are None, we don't include them.
        
        Two actions:
         - What to do with text?  Currently: mean pool
         - How to combine text and non-text representations? Currently: concat
         
        TODO update this function to produce different feature representations for different models, potentially returning multiple arguments
        """
        if source_text_arrs is None or candidate_text_arrs is None:
            # ignore text features
            # concatenate the available features
            feature_arr = np.concatenate([source_feature_arr, candidate_feature_arr, source_candidate_feature_arr])
            return feature_arr
        source_text_arr = []
        candidate_text_arr = []
        if (self.config.pool_text_feature == 0):
            source_text_arr = np.mean(source_text_arrs, axis=0)  # mean pool the available texts
            candidate_text_arr = np.mean(candidate_text_arrs, axis=0)
        elif (self.config.pool_text_feature == 1):
            source_text_arr = np.concatenate([source_text_arrs[0],source_text_arrs[1], source_text_arrs[2]])
            candidate_text_arr = np.concatenate([candidate_text_arrs[0],candidate_text_arrs[1],candidate_text_arrs[2]])
        elif (self.config.pool_text_feature == 2):
            candidate_text_arr = np.amax(candidate_text_arrs, axis=0)
            source_text_arr = np.amax(candidate_text_arrs, axis=0)
        # concatenate the available features
        feature_arr = np.concatenate([source_feature_arr, candidate_feature_arr, source_candidate_feature_arr, source_text_arr, candidate_text_arr])
        return feature_arr
    
    
    def get_input_arrs_from_triple_dicts(self, triple_dicts):
        """
        :returns - feature_arrs - a list of np.ndarrays, via get_input_arrs_from_triple_dict. Note: UNSCALED.
                 - ys - a list of y values
                 - missing_journal_id_list - a list of journal ids that prevented the construction of any input arrs
        """
        logger = logging.getLogger("cbrec.feature_loader.FeatureLoader.get_input_arrs_from_triple_dicts")
        feature_arrs = []
        ys = []
        missing_journal_id_list = []
        text_db = embeddingdb.get_text_feature_db(self.config)
        with text_db:
            n_processed = 0
            n_invalid = 0
            n_invalid_missing_texts = 0
            for triple_dict in tqdm(triple_dicts, desc='Processing triples'):
                if ((not self.config.number_of_initiation is None) and n_processed ==  self.config.number_of_initiation):
                    break
                n_processed += 1
                try:
                    source_target_arr, source_alt_arr, missing_journal_ids = self.get_input_arrs_from_triple_dict(triple_dict, text_db)
                except ValueError as ex:
                    # currently, if not enough texts available, this will result in an exception
                    n_invalid += 1
                    #logger.debug(ex)
                    continue
                if len(missing_journal_ids) > 0:
                    n_invalid_missing_texts += 1
                    missing_journal_id_list.extend(missing_journal_ids)
                    continue
                feature_arrs.append(source_target_arr)
                ys.append(1)
                feature_arrs.append(source_alt_arr)
                ys.append(0)
        if n_invalid > 0:
            logger.debug(f"After processing {n_processed} triple dicts, identified {n_invalid} invalid (and an additional {n_invalid_missing_texts} invalid due to missing text features)")
        return feature_arrs, ys, missing_journal_id_list
    
    
    def get_input_arrs_from_triple_dict(self, triple_dict, text_db, include_text=True, return_missing_journal_ids=True):
        source_usp = (triple_dict['source_user_id'], triple_dict['source_site_id'])
        target_usp = (triple_dict['target_user_id'], triple_dict['target_site_id'])
        alt_usp = (triple_dict['alt_user_id'], triple_dict['alt_site_id'])
        
        source_feature_arr = triple_dict['source_feature_arr']
        target_feature_arr = triple_dict['target_feature_arr']
        alt_feature_arr = triple_dict['alt_feature_arr']
        source_target_feature_arr = triple_dict['source_target_feature_arr']
        source_alt_feature_arr = triple_dict['source_alt_feature_arr']
            
        if include_text:
            source_journal_ids = self.journal_id_lookup.get_journal_updates_before(source_usp, triple_dict['interaction_timestamp'])
            target_journal_ids = self.journal_id_lookup.get_journal_updates_before(target_usp, triple_dict['interaction_timestamp'])
            alt_journal_ids = self.journal_id_lookup.get_journal_updates_before(alt_usp, triple_dict['interaction_timestamp'])
            if len(source_journal_ids) < 3 or len(target_journal_ids) < 3 or len(alt_journal_ids) < 3:
                raise ValueError(f"Insufficient texts: source n={len(source_journal_ids)}, target n={len(target_journal_ids)}, alt n={len(alt_journal_ids)}")
            source_text_arrs = embeddingdb.get_text_feature_arrs_from_db(text_db, source_journal_ids)
            target_text_arrs = embeddingdb.get_text_feature_arrs_from_db(text_db, target_journal_ids)
            alt_text_arrs = embeddingdb.get_text_feature_arrs_from_db(text_db, alt_journal_ids)
            
            if sum([arr is not None for arr in source_text_arrs]) < 3 or sum([arr is not None for arr in target_text_arrs]) < 3 or sum([arr is not None for arr in alt_text_arrs]) < 3:
                if return_missing_journal_ids:
                    missing_journal_id_list = []
                    for ids, texts in zip([source_journal_ids, target_journal_ids, alt_journal_ids], [source_text_arrs, target_text_arrs, alt_text_arrs]):
                        for journal_oid, text in zip(ids, texts):
                            if text is None:
                                missing_journal_id_list.append(journal_oid)
                    if len(missing_journal_id_list) > 0:
                        return None, None, missing_journal_id_list
                raise ValueError(f"Embeddings unavailable for texts (total {len(self.missing_journal_id_list)}): source n={len(source_text_arrs)}, target n={len(target_text_arrs)}, alt n={len(alt_text_arrs)}")
            
            # filter out any non-available texts            
            source_text_arrs = [arr for arr in source_text_arrs if arr is not None]
            target_text_arrs = [arr for arr in target_text_arrs if arr is not None]
            alt_text_arrs = [arr for arr in alt_text_arrs if arr is not None]
            source_target_arr = self.combine_feature_arrs(source_feature_arr, target_feature_arr, source_target_feature_arr, source_text_arrs, target_text_arrs)
            source_alt_arr = self.combine_feature_arrs(source_feature_arr, alt_feature_arr, source_alt_feature_arr, source_text_arrs, alt_text_arrs)            
        else:  # don't include text
            source_target_arr = self.combine_feature_arrs(source_feature_arr, target_feature_arr, source_target_feature_arr, None, None)
            source_alt_arr = self.combine_feature_arrs(source_feature_arr, alt_feature_arr, source_alt_feature_arr, None, None)   
        return source_target_arr, source_alt_arr, []
    
    
    def get_text_arrs(self, text_db, usp, timestamp):
        journal_ids = self.journal_id_lookup.get_journal_updates_before(usp, timestamp)
        if len(journal_ids) < 3:
            raise ValueError(f"Insufficient texts: {usp} n={len(journal_ids)}")
        # TODO offer caching option to reduce successive database calls
        text_arrs = embeddingdb.get_text_feature_arrs_from_db(text_db, journal_ids)
        text_arrs = [arr for arr in text_arrs if arr is not None]
        if len(text_arrs) < 1:  # TODO this allows us to make predictions even when all journals are not available
            raise ValueError(f"Embeddings unavailable for texts: {usp} n={len(text_arrs)}")
        return text_arrs
    
    
    def get_input_matrix_from_test_context(self, rc):
        """
        Returns UNSCALED feature matrices for a given RecContext
        """
        if rc.metadata_id in self.rec_input_matrix_cache:
            return self.rec_input_matrix_cache[rc.metadata_id]
        arrs = []
        text_db = embeddingdb.get_text_feature_db(self.config)
        with text_db:
            for i in range(len(rc.source_usp_mat)):
                source_feature_arr = rc.source_usp_mat[i,:]
                source_usp = (rc.source_usp_arr[i,0], rc.source_usp_arr[i,1])
                source_text_arrs = self.get_text_arrs(text_db, source_usp, rc.timestamp)
                for j in range(len(rc.candidate_usp_mat)):
                    candidate_feature_arr = rc.candidate_usp_mat[j,:]

                    ind = (i * len(rc.candidate_usp_arr)) + j
                    source_candidate_feature_arr = rc.user_pair_mat[ind,:]
                    
                    candidate_usp = (rc.candidate_usp_arr[j,0], rc.candidate_usp_arr[j,1])
                    candidate_text_arrs = self.get_text_arrs(text_db, candidate_usp, rc.timestamp)
                    
                    arr = self.combine_feature_arrs(source_feature_arr, candidate_feature_arr, source_candidate_feature_arr, source_text_arrs, candidate_text_arrs)
                    arrs.append(arr)
        X = np.vstack(arrs)
        self.rec_input_matrix_cache[rc.metadata_id] = X
        return X
        