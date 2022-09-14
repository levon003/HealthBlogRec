
import numpy as np

import cbrec.evaluation
import cbrec.reccontext
import cbrec.genconfig
import cbrec.coverage
import cbrec.modeling.modelconfig
import cbrec.modeling.manager


class ModelScorer(cbrec.evaluation.Scorer):
    """
    Given a ModelManager, can compute scores and metrics for a test RecContext.
    """
    def __init__(self, 
                 config: cbrec.genconfig.Config,
                 test_context: cbrec.reccontext.RecContext, 
                 model_manager: cbrec.modeling.manager.ModelManager, 
                 coverage_tracker: cbrec.coverage.CoverageTracker=None, 
                 save_scores: bool = True,
            ):
        super().__init__(config, test_context, coverage_tracker=coverage_tracker, save_scores=save_scores)
        self.model_manager = model_manager
        
        # extract the model name from the manager's config
        self.model_name = self.model_manager.model_config.model_name

    
    def score(self):
        """
        Score using associated model.
        """
        X = self.test_context.X_test
        y_score = self.model_manager.score_test_matrix(X)
        
        y_score_mat = self.get_empty_score_arr('full')
        y_score_mat = y_score.reshape((y_score_mat.shape[1], y_score_mat.shape[0])).T

        y_score_site = self.reduce_usp_ranking_to_site(self.merge_multisource_rankings(y_score_mat))
        self.compute_metrics(y_score_site, model_name=self.model_name)

        if len(self.target_site_id_inds) > 0:
            # produce a bonus metric: accuracy on this test context
            y_pred = (y_score_site >= 0.5).astype(int)
            acc = np.sum(y_pred == self.y_true) / self.y_true.shape[1]
            self.metrics_dict[self.model_name]['acc'] = acc
        
        return self.metrics_dict[self.model_name]