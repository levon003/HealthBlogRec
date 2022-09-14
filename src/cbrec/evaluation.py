"""
Module for evaluation and storing.  See also coverage module.

Evaluation is based around a few things:
-Which SITES are shown to the user
-Which RECOMMENDED USERS reciprocate?

Evaluation metrics:

pred_coverage: at each timestep, what is the number of active&eligible authors vs active&existing authors?  (What percent of active authors can we actually generate recs for?)
coverage@n: wehat % of sites with eligible authors appear in the top n recs at any time during the test period?  even better: should count HOW MANY times each site is recced, and generate a distribution.

For BPR models:
mean_target_score: take the mean of the raw score produced by the model for all test initiations.  (Want this to be as close to 1 as possible, and can be compared to the training period to determine how much of a test drop-off there is.)
    Specifically, compute the following mean scores: (training, train_usp_source/train_usp_target, test_usp_source/train_usp_target, train_usp_source/test_usp_target, test_usp_source/test_usp_target)  A USP is in training if either the user or the site existed pre-2018.

For each test interaction, save the following:
model_name
n_usps_scored
n_sites_scored
source_target_raw_score
rank_of_target
reciprocal_rank  # reciprocal_rank = 1 / rank_of_target
ndcg_1  # note: ndcg is computed with y_true = [1, 0, 0, ...] and y_score = the raw scores, with the scores being 
ndcg_5
ndcg_10
ndcg_50

Maybe:
n_target_users
source_target_backup_scores
source_target_backup_ranks  # the pre-deduplication ranks if the backup score
usp_source_target_raw_score
usp_rank_of_target  # the rank of the target pre-deduplication
The idea is that we first score all eligible&active USPs.  Then, identify duplicate site_ids and resolve according to strategy (e.g. take max).
Do we then have a separate combination step for multiple sources, or do we generate multiple summary lines if there are multiple source sites (and thus USPs)?

Some of the following may be needed to compute coverage: (But: maybe not, since coverage seems like a feature of the USER (or perhaps the SITE or USP, but intuitively we care less about that).)
n_active_eligible_usps
n_active_eligible_sites
n_eligible_usps
n_eligible_sites
n_active_existing_usps
n_active_existing_sites


What are the baselines?

By our somewhat idiosyncratic definition, baselines are any scoring model that doesn't require training.  

Baselines can be site-centric (directly producing a scoring of sites) or usp-centric (producing a scoring of usps that will be reduced to a site scoring).
Additionally, baselines can be non-personalized (ignoring any source_usps) or personalized (producing a scoring for each source usp that needs to be merged and reduced).

Non-personalized models:
MostRecentJournal: Rank eligible authors according to recency of journal updates
MostJournals: Rank eligible authors according to total number of journal updates (within some time range)
MostRecentInitiation: Rank eligible authors according to recency of being initiated with by others
MostInitiations: Rank eligible authors according to count of being initiated with by others (within some time range)

Personalized models:
ClosestToStart: Rank eligible authors according to temporal proximity between first journal update
CosineSimilarity: Rank eligible authors according to cosine similarity of features
BPR
    Activity features only
    Author role only
    Network features only
    Journal embeddings only
    All features
    
    Underlying model:
    Cosine similarity (not trained: use raw cosine similarity as the score. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
        One thought: maybe cosine similarity should actually be a shared feature?  After all, learning the dot product is notoriously hard...
    Logistic regression (with potentially multiple feature inputs: e.g. concat [a,b,ab], [a-b,ab] [a*b,ab])
    Something from the BPR paper?
    Neural something?

A ranking is a list of USPs.  Or rather, a list of sites?

"""

from . import featuredb
from . import coverage
from . import reccontext
from . import torchmodel

import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import rankdata


class Scorer:
    """
    Given a RecContext, this class provides functions for scoring in that context and producing metrics for the produced scores.

    test_context is expected to have a candidate_usp_arr -- a score array with dim [X, 2] that contains the user/site pairs. Needs to be sorted by site_id.
    Other test_context requirements:
        is_test_period == True
        is_initiation_eligible == True
        source_user_id
        target_site_id
        candidate_usp_arr
        source_usp_arr
        source_usp_mat
        candidate_usp_mat
        user_pair_mat
    """
    def __init__(self, config, test_context: reccontext.RecContext, coverage_tracker: coverage.CoverageTracker=None, save_scores: bool = False):
        self.config = config
        if not test_context.is_test_period or not test_context.is_initiation_eligible:
            raise ValueError("Bad RecContext; expected test context.")
        self.test_context = test_context

        # we compute the site ids for which scores will be generated (after merging and reduction)
        site_id_arr, indices = np.unique(test_context.candidate_usp_arr[:,1], return_index=True)
        self.site_id_arr = site_id_arr  # final scores are produced with an order corresponding to site_id_arr
        self.candidate_usp_site_id_indices = indices

        # identify which indices in site_id_arr represent target inds
        self.target_site_id_inds = np.argwhere(self.site_id_arr == test_context.target_site_id)

        # we generate the y_true against which metrics will be computed
        # TODO This process currently uses a single target site as the ground truth, but we should probably be able to consider multiple options (e.g. to consider multiple sites as valid recommendations, or to give them different weights in y_true)
        self.y_true = np.zeros(len(self.site_id_arr), dtype=np.int8)
        if len(self.test_context.target_inds) > 0:
            self.y_true[self.target_site_id_inds] = 1
        self.y_true = self.y_true.reshape(1, -1)

        self.coverage_tracker = coverage_tracker

        self.save_scores = save_scores
        if self.save_scores:
            self.scores_dict = {}  # map of model_name -> y_score_site array

        self.metrics_dict = {}  # map of model_name -> metrics

    def get_empty_score_arr(self, array_type):
        if array_type == 'full':
            # full score array, dim candidate_usps X source_usps
            return np.empty((len(self.test_context.candidate_usp_arr), len(self.test_context.source_usp_arr)), dtype=featuredb.NUMPY_DTYPE)
        elif array_type == 'merged':
            # single-usp (merged) score array of usps, dim candidate_usps X 1
            return np.empty(len(self.test_context.candidate_usp_arr), dtype=featuredb.NUMPY_DTYPE)
        elif array_type == 'reduced':
            # score array of sites (with implied merger and reduction), dim site_ids X 1
            return np.empty(len(self.site_id_arr), dtype=featuredb.NUMPY_DTYPE)

    def merge_multisource_rankings(self, y_score_mat):
        """
        :y_score_mat -- Should have dim len(candidate_usps) X len(source_usps)

        :return -- Returns y_score_usp with dim (len(candidate_usps),)
        """
        merge_strategy = 'max'
        if merge_strategy == 'max':
            y_score_usp = np.max(y_score_mat, axis=1)
        else:
            raise ValueError(f"Merge strategy {merge_strategy} not yet implemented.")
        return y_score_usp

    def reduce_usp_ranking_to_site(self, y_score_usp):
        """
        :y_score_usp -- score array with dim candidate_usps.shape[0]

        :returns
        y_score_site -- array of scores, corresponding to the sites in site_id_arr

        """
        merge_strategy = 'max'
        if merge_strategy == 'max':
            y_score_site = np.maximum.reduceat(y_score_usp, self.candidate_usp_site_id_indices)
        else:
            raise ValueError(f"Merge strategy {merge_strategy} not yet implemented.")
        return y_score_site

    def compute_ranks(self, y_score_site):
        """
        Rank sites by score, assuming the highest score should have rank 1 and the lowest should have rank n.
        Break-ties by assinging the lowest rank e.g. rank y_score_site = [1.0, 0.5, 0.5, 0.0] as ranks = [1, 3, 3, 4]
        """
        return rankdata(-1 * y_score_site, method='max')

    def compute_metrics(self, y_score_site, model_name=""):
        """
        Computes some summary metrics.

        If target_site_id_inds exist, additional metrics are computed.
        """
        if self.save_scores: 
            self.scores_dict[model_name] = y_score_site

        # actually compute metrics
        metric_dict, ranks = self.compute_nontarget_metrics(y_score_site)
        if len(self.target_site_id_inds) > 0:
            metric_dict_target = self.compute_target_metrics(y_score_site, ranks)
            metric_dict.update(metric_dict_target)

        # save the metric dict, and update coverage if appropriate
        self.metrics_dict[model_name] = metric_dict
        if self.coverage_tracker: 
            self.coverage_tracker.register_ranking(self.site_id_arr, ranks, model_name=model_name)
        
        return metric_dict, ranks

    def compute_nontarget_metrics(self, y_score_site):
        assert y_score_site.shape == self.site_id_arr.shape, y_score_site.shape
        assert np.all(np.isfinite(y_score_site)), "Expected entirely finite y_score_site arr." #str(y_score_site[~np.isfinite(y_score_site)])
        sort_inds = np.argsort(y_score_site)

        lowest_scores = y_score_site[sort_inds[:3]]
        lowest_score_site_ids = self.site_id_arr[sort_inds[:3]]

        highest_scores = y_score_site[sort_inds[-3:]]
        highest_score_site_ids = self.site_id_arr[sort_inds[-3:]]

        ranks = self.compute_ranks(y_score_site)

        metric_dict = {
            'n': len(y_score_site),
            'lowest_scores': lowest_scores.tolist(),
            'lowest_score_site_ids': lowest_score_site_ids.tolist(),
            'highest_scores': highest_scores.tolist(),
            'highest_score_site_ids': highest_score_site_ids.tolist(),
        }
        return metric_dict, ranks

    def compute_target_metrics(self, y_score_site, ranks):
        #assert len(self.target_site_id_inds) >= 1, "No target inds! Can't compute metrics."
        #rank_ind = np.argwhere(sort_inds == self.target_site_id_inds)[0,0]
        #target_raw_score = y_score_site[rank_ind]
        #rank_of_target = len(y_score_site) - rank_ind

        # in the case of multiple target sites, we take the best score (the highest) and the best rank (the lowest)
        target_raw_score = max([y_score_site[ind] for ind in self.target_site_id_inds])
        rank_of_target = min([ranks[ind] for ind in self.target_site_id_inds])
        
        #ndcg_scores = []
        #y_score_site_column = y_score_site.reshape(1, -1)
        #for k in [1, 5, 10, 50]:
        #    ndcg = ndcg_score(self.y_true, y_score_site_column, k=k)
        #    ndcg_scores.append(ndcg)

        metric_dict = {
            'target_raw_score': float(target_raw_score),
            'target_rank': int(rank_of_target),
            #'ndcg_1': float(ndcg_scores[0]),
            #'ndcg_5': float(ndcg_scores[1]),
            #'ndcg_10': float(ndcg_scores[2]),
            #'ndcg_50': float(ndcg_scores[3]),
        }
        return metric_dict


class SklearnModelScorer(Scorer):
    """
    Given an sklearn model with the predict_proba function, can compute scores and metrics for a test RecContext.
    """
    def __init__(self, config, test_context: reccontext.RecContext, 
                clf, model_name="SklearnModel",
                coverage_tracker: coverage.CoverageTracker=None, save_scores: bool = False
            ):
        super().__init__(config, test_context, coverage_tracker=coverage_tracker, save_scores=save_scores)
        self.clf = clf
        self.model_name = model_name

    def get_design_matrix(self):
        rc = self.test_context  # alias for code brevity
        arrs = []
        for i in range(len(rc.source_usp_mat)):
            source_feature_arr = self.test_context.source_usp_mat[i,:]
            for j in range(len(rc.candidate_usp_mat)):
                candidate_feature_arr = rc.candidate_usp_mat[j,:]
                
                ind = (i * len(rc.candidate_usp_arr)) + j
                source_candidate_feature_arr = rc.user_pair_mat[ind,:]
                arr = np.concatenate([source_feature_arr, candidate_feature_arr, source_feature_arr - candidate_feature_arr, source_candidate_feature_arr])
                arrs.append(arr)
        X = np.vstack(arrs)
        return X

    def score_proba(self):
        """
        Score using the predict_proba method of the passed classifier.
        TODO could also implement one of these that calls decision_function instead.
        """
        X = self.get_design_matrix()
        y_score = self.clf.predict_proba(X)[:,1]

        y_score_mat = self.get_empty_score_arr('full')
        y_score_mat = y_score.reshape((y_score_mat.shape[1], y_score_mat.shape[0])).T

        y_score_site = self.reduce_usp_ranking_to_site(self.merge_multisource_rankings(y_score_mat))
        self.compute_metrics(y_score_site, model_name=self.model_name)

        return self.metrics_dict[self.model_name]
    
class TorchModelScorer(Scorer):
    """
    Given a PyTorch model, can compute scores and metrics for a test RecContext.
    """
    def __init__(self, config, test_context: reccontext.RecContext, 
                torch_model: torchmodel.TorchModel, model_name="PytorchModel",
                coverage_tracker: coverage.CoverageTracker=None, save_scores: bool = False
            ):
        super().__init__(config, test_context, coverage_tracker=coverage_tracker, save_scores=save_scores)
        self.torch_model = torch_model
        self.model_name = model_name

    def score(self):
        """
        Score using associated torch model.
        """
        X = self.test_context.X_test  # self.torch_model.get_input_matrix_from_test_context(self.test_context)
        y_score = self.torch_model.score_test_matrix(X)
        
        y_score_mat = self.get_empty_score_arr('full')
        y_score_mat = y_score.reshape((y_score_mat.shape[1], y_score_mat.shape[0])).T

        y_score_site = self.reduce_usp_ranking_to_site(self.merge_multisource_rankings(y_score_mat))
        self.compute_metrics(y_score_site, model_name=self.model_name)

        if len(self.target_site_id_inds) > 0:
            # produce a bonus metric: accuracy on this test context
            # later note: I believe this implementation is broken, due to the shape of self.y_true; however, the TorchModelScorer is essentially deprecated
            y_pred = (y_score_site >= 0.5).astype(int)
            acc = np.sum(y_pred == self.y_true) / len(self.y_true)
            self.metrics_dict[self.model_name]['acc'] = acc
        
        return self.metrics_dict[self.model_name]
        

class BaselineScorer(Scorer):
    def __init__(self, config, test_context: reccontext.RecContext, coverage_tracker: coverage.CoverageTracker=None, save_scores: bool = False):
        super().__init__(config, test_context, coverage_tracker=coverage_tracker, save_scores=save_scores)

    def compute_baselines(self, activity_manager):
        """
        Adds a 'baseline_metrics' entry to the associated test_context metadata dict.

        In addition to the existing features, also need the activity_manager to compute all baselines.
        """

        self.test_context.md['baseline_metrics'] = self.metrics_dict

        # produce scores for the baselines that generate scores for all source/candidate usp pairs
        for model_name, y_score_mat in [
            ('NaiveNetwork', self.compute_NaiveNetwork()),
            ('CosineSimilarity', self.compute_CosineSimilarity()),
        ]:
            assert np.all(np.isfinite(y_score_mat)), f"{model_name} {y_score_mat[~np.isfinite(y_score_mat)]}"
            y_score_site = self.reduce_usp_ranking_to_site(self.merge_multisource_rankings(y_score_mat))
            self.compute_metrics(y_score_site, model_name)

        # produce scores for the baselines that generate scores for all candidate usp pairs
        for model_name, y_score_usp in [
            ('MostInitiatedWith', self.compute_MostInitiatedWith()),
            ('ClosestToStart', self.compute_ClosestToStart(activity_manager)),
        ]:
            y_score_site = self.reduce_usp_ranking_to_site(y_score_usp)
            self.compute_metrics(y_score_site, model_name)

        # produce scores for the baselines that generate scores for all sites
        y_score_site_count, y_score_site_recent = self.compute_MostInitiatedWithRecently(activity_manager)
        self.compute_metrics(y_score_site_count, 'MostInitiatedWithRecently')
        self.compute_metrics(y_score_site_recent, 'MostRecentlyInitiatedWith')

        y_score_site_count, y_score_site_recent = self.compute_MostJournalsRecently(activity_manager)
        self.compute_metrics(y_score_site_count, 'MostJournalsRecently')
        self.compute_metrics(y_score_site_recent, 'MostRecentJournal')

        # Random baseline
        y_score_site_random = self.get_empty_score_arr('reduced')
        y_score_site_random = self.config.rng.uniform(0, 1, size=y_score_site_random.shape)
        self.compute_metrics(y_score_site_random, 'Random')

    def compute_NaiveNetwork(self):
        """

        Non-vectorized code that demonstrates the algorithm:
        for j in range(y_score_mat.shape[1]):  # each column is a source_usp
            for i in range(y_score_mat.shape[0]):  # each row is a candidate usp
                network_features = self.test_context.user_pair_mat[(i*len(y_score_mat.shape[1]))+j,0:2]
                if network_features[2] == 1.0:  # is_reciprocal
                    score = 3.0
                elif network_features[1] == 1.0:  # is_fof
                    score = 2.0
                elif network_features[0] == 1.0:  # are_weakly_connected
                    score = 1.0
                else:
                    score = 0.0
                y_score_mat[i,j] = score

        The vectorized code exploits the fact that the features are additive, i.e. if is_reciprocal is 1 than is_fof and are_weakly_connected must also be 1.

        """
        y_score_mat = self.get_empty_score_arr('full')
        for j in range(y_score_mat.shape[1]):  # for each source_usp
            start_ind = j * len(self.test_context.candidate_usp_arr)
            stop_ind = start_ind + len(self.test_context.candidate_usp_arr)
            user_feats = self.test_context.user_pair_mat[start_ind:stop_ind,0:3]
            y_score_mat[:,j] = user_feats.sum(axis=1)
            #if not np.all(y_score_mat[:,j] <= 3.0):
            #    for i in range(user_feats.shape[0]):
            #        print(user_feats[i,:], y_score_mat[i,j], (np.sum(user_feats[i,:]) > 3) or (np.sum(user_feats[i,:]) < 0))
            #    print(self.test_context.user_pair_mat.shape, start_ind, stop_ind)
            #    print(user_feats.shape)
            #    print(f"{y_score_mat[:,j].shape} {user_feats.sum(axis=1).shape} {user_feats.sum(axis=1)}")
            #assert np.all(np.isfinite(y_score_mat[:,j])), y_score_mat[:,j]
        return y_score_mat

    def compute_CosineSimilarity(self):
        """
        :returns
            y_score_mat -- full 
        """
        y_score_mat = self.get_empty_score_arr('full')
        sim_mat = cosine_similarity(self.test_context.candidate_usp_mat, self.test_context.source_usp_mat)
        # entry i, j corresponds to the similarity between candidate_usp_mat[i,:] and source_usp_mat[j,:]
        y_score_mat = sim_mat
        return y_score_mat

    def compute_ClosestToStart(self, ram):
        """

        :ram -- RecentActivityManager

        :returns
            y_score_usp -- difference between the start time of the source user and each candidate usp. 
        """
        y_score_usp = self.get_empty_score_arr('merged')
        user_ids = self.test_context.candidate_usp_arr[:,0]
        for i, user_id in enumerate(user_ids):
            first_journal_timestamp = ram.get_first_journal_update_timestamp(user_id)
            #assert first_journal_timestamp is not None, f"User {user_id} had no recorded first journal update timestamp."
            first_journal_timestamp = first_journal_timestamp / self.config.ms_per_hour if first_journal_timestamp is not None else np.finfo(featuredb.NUMPY_DTYPE).max
            y_score_usp[i] = first_journal_timestamp
        y_score_usp -= ram.get_first_journal_update_timestamp(self.test_context.source_user_id) / self.config.ms_per_hour
        # take the absolute value and make it negative, such that the smallest absolute differences are the largest values in the array i.e. 0 is best 
        y_score_usp = np.abs(y_score_usp) * -1
        
        return y_score_usp

    def compute_MostInitiatedWith(self):
        indegree = self.test_context.candidate_usp_mat[:,0]
        y_score_usp = indegree
        return y_score_usp

    def compute_MostInitiatedWithRecently(self, activity_manager):
        """
        Computes both MostInitiatedWithRecently and MostRecentlyInitiatedWith
        """
        rac = activity_manager.get_activity_counter('initiation_site')
        y_score_site_count, y_score_site_recent = self.get_scores_from_site_counter(rac)
        return y_score_site_count, y_score_site_recent

    def compute_MostJournalsRecently(self, activity_manager):
        """
        Computes both MostJournalsRecently and MostRecentJournal
        """
        rac = activity_manager.get_activity_counter('journal_site')
        y_score_usp_count, y_score_usp_recent = self.get_scores_from_site_counter(rac)
        return y_score_usp_count, y_score_usp_recent

    def get_scores_from_site_counter(self, rac):
        """
        Generate y_score_site arrays from the given RecentActivityCounter, which is assumed to be tracking site_ids.

        :rac -- recentActivityCounter.RecentActivityCounter

        :returns
            y_score_site_count -- count of recent activity
            y_score_site_recent -- number of seconds to current timestamp
        """
        y_score_site_count = self.get_empty_score_arr('reduced')
        y_score_site_recent = self.get_empty_score_arr('reduced')
        no_recent_score = self.test_context.timestamp / self.config.ms_per_hour
        for i, site_id in enumerate(self.site_id_arr):
            n_recent = rac.get_count(site_id)
            if n_recent > 0:
                most_recent = rac.get_most_recent_activity(site_id)
                time_to_most_recent = self.test_context.timestamp - most_recent
                # convert difference from ms to hours
                time_to_most_recent /= self.config.ms_per_hour
                #time_to_most_recent *= -1  # invert most recent, so that the highest possible value is 0 and the lowest possible value is self.test_context.timestamp
            else:
                time_to_most_recent = no_recent_score

            y_score_site_count[i] = n_recent
            y_score_site_recent[i] = time_to_most_recent
        y_score_site_recent *= -1
        return y_score_site_count, y_score_site_recent
