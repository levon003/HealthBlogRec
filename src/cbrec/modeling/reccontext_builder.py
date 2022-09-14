
import numpy as np

import cbrec.reccontext

def build_reccontext(config, text_loader, md, test_context):
    """
    This builds a RecContext, populating the matrix X_test using the given text_loader.
    """
    rc = cbrec.reccontext.RecContext.create_from_test_context(config, md, test_context)
    build_reccontext_X(config, text_loader, rc)
    return rc


def build_reccontext_X(config, text_loader, rc):
    # build X_test
    n_source_usps = len(rc.source_usp_mat)
    n_candidate_usps = len(rc.candidate_usp_mat)
    n_rows = n_source_usps * n_candidate_usps
    assert n_rows == len(rc.user_pair_mat)
    n_cols = 2 * (config.user_feature_count + config.text_feature_count) + config.user_pair_feature_count
    X = np.empty((n_rows, n_cols), dtype=np.float32)
    
    source_feature_arr_end = config.user_feature_count
    candidate_feature_arr_end = source_feature_arr_end + config.user_feature_count
    source_candidate_feature_arr_end = candidate_feature_arr_end + config.user_pair_feature_count
    source_text_arr_end = source_candidate_feature_arr_end + config.text_feature_count
    candidate_text_arr_end = source_text_arr_end + config.text_feature_count
    assert candidate_text_arr_end == n_cols
    
    # start by copying in the user_pair_mat features
    X[:,candidate_feature_arr_end:source_candidate_feature_arr_end] = rc.user_pair_mat
    
    # then, copy in the source USP info
    for i in range(n_source_usps):
        source_feature_arr = rc.source_usp_mat[i,:]
        source_usp = tuple(rc.source_usp_arr[i,:])
        source_text_arr = text_loader.get_text_features(source_usp, rc.timestamp)
        
        row_start = i * n_candidate_usps
        row_end = row_start + n_candidate_usps
        X[row_start:row_end,0:source_feature_arr_end] = source_feature_arr
        X[row_start:row_end,source_candidate_feature_arr_end:source_text_arr_end] = source_text_arr
        
        # copy in the candiate USP feature info
        assert rc.candidate_usp_mat.shape == (row_end - row_start, candidate_feature_arr_end - source_feature_arr_end)
        X[row_start:row_end,source_feature_arr_end:candidate_feature_arr_end] = rc.candidate_usp_mat
    
    # then, copy in the candidate USP info
    # note we only need to copy the texts, since the candidate_usp_mat was already copied in above
    
    # we use row_inds to keep track of which rows are associated with each candidate
    # e.g. if there are two source usps, then the first candidate has data on row 0 and row n, where n is the number of candidate usps
    row_inds = np.array([(i * n_candidate_usps) for i in range(n_source_usps)])
    bad_row_inds = []
    for j in range(n_candidate_usps):
        #candidate_feature_arr = rc.candidate_usp_mat[j,:]
        candidate_usp = tuple(rc.candidate_usp_arr[j,:])
        try:
            candidate_text_arr = text_loader.get_text_features(candidate_usp, rc.timestamp)
            X[row_inds,source_text_arr_end:candidate_text_arr_end] = candidate_text_arr
        except ValueError:
            # insufficient texts...
            bad_row_inds.extend(row_inds)

        #X[row_inds,source_feature_arr_end:candidate_feature_arr_end] = candidate_feature_arr
        row_inds += 1
        
        # this is the row-level computation for the row_inds
        #row_inds = np.array([(i * n_candidate_usps) + j for i in range(n_source_usps)])
    if len(bad_row_inds) > 0:
        # this is a very bad case; candidates in the generated USP are missing data
        # so we need to drop those candidates, but even worse we need to 
        bad_row_inds = np.array(bad_row_inds)
        X = np.delete(X, bad_row_inds, axis=0)
        # we update the list of candidate USPs included in the model
        if len(rc.target_inds) > 0:
            for i, target_ind in enumerate(rc.target_inds):
                if target_ind in bad_row_inds:
                    raise ValueError("Target USP has missing data; unrecoverable case for testing.")
                # we're about to permanently adjust the candidate_usp_arr, 
                # so we need to account for the rows we're deleting and update the target_inds to still point to the appropriate rows
                rc.target_inds[i] -= (bad_row_inds < target_ind).sum()  
        rc.candidate_usp_arr = np.delete(rc.candidate_usp_arr, bad_row_inds[bad_row_inds < rc.candidate_usp_arr.shape[0]], axis=0)
    rc.X_test = X


def combine_feature_arrs(source_feature_arr, candidate_feature_arr, source_candidate_feature_arr, source_text_arr=None, candidate_text_arr=None):
    """
    This function creates a fixed-length input to a model, given non-text feature arrs and text embeddings.

    If either of the text features are None, we don't include them.
    
    This function is not currently used, and should not be used in favor of build_reccontext_X
    """
    if source_text_arrs is None or candidate_text_arrs is None:
        # ignore text features
        # concatenate the available features
        feature_arr = np.concatenate([source_feature_arr, candidate_feature_arr, source_candidate_feature_arr])
        return feature_arr
    # concatenate the available features
    feature_arr = np.concatenate([source_feature_arr, candidate_feature_arr, source_candidate_feature_arr, source_text_arr, candidate_text_arr])
    return feature_arr