import numpy as np
import logging
import pickle
import os


class CoverageTracker:
    """
    A coverage tracker tracks the ranks of a site in one or more models.

    For memory reasons, full rankings are not necessarily saved.

    Note: streaming mean and variance are computed per https://math.stackexchange.com/a/116344/583256

    >>> import cbrec.genconfig
    >>> config = cbrec.genconfig.Config()
    >>> site_id_arr = np.arange(0, 2000)
    >>> ranks = site_id_arr + 1
    >>> ranks
    array([   1,    2,    3, ..., 1998, 1999, 2000])

    >>> ct = CoverageTracker(config, low_memory=True)
    >>> ct.register_ranking(site_id_arr, ranks)
    >>> ct.finalize_stats()
    >>> stats = ct.get_stats_by_site_id()
    >>> assert len(stats) == 2000
    >>> stats[0]
    {'n': 1, 'mean': 1, 'var': 0, 'best': 1, 'worst': 1, 'n_top_5': 1, 'n_top_100': 1, 'n_top_1000': 1}
    >>> stats[999]
    {'n': 1, 'mean': 1000, 'var': 0, 'best': 1000, 'worst': 1000, 'n_top_5': 0, 'n_top_100': 0, 'n_top_1000': 1}
    >>> stats[1999]
    {'n': 1, 'mean': 2000, 'var': 0, 'best': 2000, 'worst': 2000, 'n_top_5': 0, 'n_top_100': 0, 'n_top_1000': 0}

    >>> rng = np.random.default_rng(872)
    >>> ct = CoverageTracker(config, low_memory=True)
    >>> ranks_list = []
    >>> for i in range(10):
    ...     rng.shuffle(ranks)
    ...     ranks_list.append(ranks.copy())
    ...     ct.register_ranking(site_id_arr, ranks)
    >>> ct.finalize_stats()
    >>> stats = ct.get_stats_by_site_id()
    >>> assert len(stats) == 2000
    >>> stats[0]
    {'n': 10, 'mean': 972.1999999999999, 'var': 213708.84444444446, 'best': 420, 'worst': 1854, 'n_top_5': 0, 'n_top_100': 0, 'n_top_1000': 6}
    >>> stats[999]
    {'n': 10, 'mean': 1074.4, 'var': 227651.1555555556, 'best': 266, 'worst': 1862, 'n_top_5': 0, 'n_top_100': 0, 'n_top_1000': 3}
    >>> stats[1999]
    {'n': 10, 'mean': 825.3000000000001, 'var': 199297.7888888889, 'best': 27, 'worst': 1479, 'n_top_5': 0, 'n_top_100': 1, 'n_top_1000': 6}

    >>> rng = np.random.default_rng(872)
    >>> hm_ct = CoverageTracker(config, low_memory=False)
    >>> for ranks in ranks_list:
    ...     hm_ct.register_ranking(site_id_arr, ranks)
    >>> hm_ct.finalize_stats()
    >>> hm_stats = hm_ct.get_stats_by_site_id()
    >>> assert len(hm_stats) == 2000
    >>> var_diffs = []
    >>> for i in range(2000):
    ...     for key in stats[i]:
    ...         if key == 'var':
    ...             var_diffs.append(stats[i][key] - hm_stats[i][key])
    ...             continue
    ...         if not np.isclose(stats[i][key], hm_stats[i][key]):
    ...             print(f"{key} {stats[i][key]} {hm_stats[i][key]}")
    >>> print(f"{np.mean(var_diffs):.2f}")  # low memory variance estimates are very bad...
    33377.21
    """
    def __init__(self, config, low_memory=False):
        self.config = config
        self.logger = logging.getLogger("cbrec.coverage.CoverageTracker")
        self.low_memory = low_memory
        self.rank_dicts = {}  # map of model_name -> dict( site_id -> [integer ranks...] )

        # Dict of model_name -> dict(
        #     site_id -> dict(
        #         n
        #         mean
        #         var
        #         best
        #         worst
        #         n_top_*
        #     )
        # )
        self.site_id_stat_maps = {}
        
        self.top_n_vals = [5, 100, 1000]
        # max number of ranks to store for a given (model_name, site_id) pair
        self.max_stored_ranks = 5000  # during regular execution, this is 9 models X 10000 sites X max_stored_ranks integers to store

    def get_site_id_stat_map(self, model_name):
        if model_name not in self.site_id_stat_maps:
            self.site_id_stat_maps[model_name] = {}
        return self.site_id_stat_maps[model_name]

    def register_ranking(self, site_id_arr, ranks, model_name=""):
        if model_name not in self.rank_dicts:
            self.rank_dicts[model_name] = {}
        rank_dict = self.rank_dicts[model_name]

        if self.low_memory:
            for site_id, rank in zip(site_id_arr, ranks):
                site_id_stat_map = self.get_site_id_stat_map(model_name)

                if site_id not in site_id_stat_map:
                    stat_map = {
                        'n': 1,
                        'mean': rank,
                        'var': 0,
                        'best': rank,
                        'worst': rank,
                    }
                    stat_map.update({
                        'n_top_' + str(n_val): 1 if rank <= n_val else 0
                        for n_val in self.top_n_vals
                    })
                    site_id_stat_map[site_id] = stat_map
                else:
                    stat_map = site_id_stat_map[site_id]
                    stat_map['n'] += 1
                    n = stat_map['n']

                    # update max and min
                    if rank < stat_map['best']:
                        stat_map['best'] = rank
                    if rank > stat_map['worst']:
                        stat_map['worst'] = rank

                    # update mean and variance
                    m = stat_map['mean']
                    v = stat_map['var']
                    new_m = m + (rank - m) / n
                    new_v = v + (rank - m) * (rank - new_m)
                    stat_map['mean'] = new_m
                    stat_map['var'] = new_v

                    # update top_n counts
                    for n_val in self.top_n_vals:
                        if rank <= n_val:
                            stat_map['n_top_' + str(n_val)] += 1
        else:  # not low memory, store the raw ranks
            for site_id, rank in zip(site_id_arr, ranks):
                if site_id in rank_dict:
                    rank_dict[site_id].append(rank)
                    if len(rank_dict[site_id]) > self.max_stored_ranks:
                        site_id_stat_map = self.get_site_id_stat_map(model_name)
                        self.__update_site_stat_map(rank_dict[site_id], site_id, site_id_stat_map)
                        rank_dict[site_id] = []  # clear the list of ranks
                else:
                    rank_dict[site_id] = [rank,]

    def finalize_stats(self, model_name=None):
        """
        Currently, finalizes only the variance; everything else is already good to go.

        If model_name is not provided, finalizes stats for all model names.
        """
        if not self.low_memory:
            return

        if model_name is None:
            for model_name in self.site_id_stat_maps.keys():
                self.finalize_stats(model_name=model_name)
        else:
            site_id_stat_map = self.site_id_stat_maps[model_name]
            for stat_map in site_id_stat_map.values():
                # finalize variance
                if stat_map['n'] > 1:
                    stat_map['var'] = stat_map['var'] / (stat_map['n'] - 1)

    def get_stats_by_site_id(self, model_name=""):
        if self.low_memory:
            site_id_stat_map = self.site_id_stat_maps[model_name]
            return site_id_stat_map
        else:
            # need to compute site coverage counts
            # (or retrieve cached counts)
            return self.compute_site_coverage_stats(model_name=model_name)        

    def __update_site_stat_map(self, ranks, site_id, site_id_stat_map):
        if len(ranks) == 0:
            # no updates necessary
            return
        ranks = np.array(ranks)
        n = len(ranks)
        mean = np.mean(ranks)
        var = np.var(ranks)
        best = np.min(ranks)
        worst = np.max(ranks)
        if site_id not in site_id_stat_map:
            stat_map = {
                'n': n,
                'mean': mean,
                'var': var,
                'best': best,
                'worst': worst,
            }
            stat_map.update({
                'n_top_' + str(n_val): np.sum(ranks <= n_val)
                for n_val in self.top_n_vals
            })
            site_id_stat_map[site_id] = stat_map
        else:
            stat_map = site_id_stat_map[site_id]
            old_n = stat_map['n']
            stat_map['n'] = old_n + n
            stat_map['mean'] = (stat_map['mean'] * old_n + mean * n) / (old_n + n)
            stat_map['var'] = (stat_map['var'] * old_n + var * n) / (old_n + n)
            stat_map['best'] = min(stat_map['best'], best)
            stat_map['worst'] = max(stat_map['worst'], worst)
            stat_map.update({
                'n_top_' + str(n_val): stat_map['n_top_' + str(n_val)] + np.sum(ranks <= n_val)
                for n_val in self.top_n_vals
            })

    def compute_site_coverage_stats(self, model_name=""):
        if self.low_memory:
            raise ValueError("Only run these computations when not low memory.")
        rank_dict = self.rank_dicts[model_name]
        site_id_stat_map = self.get_site_id_stat_map(model_name)

        for site_id in rank_dict.keys():
            ranks = rank_dict[site_id]
            self.__update_site_stat_map(ranks, site_id, site_id_stat_map)
            rank_dict[site_id] = []  # clear the list of ranks
        return site_id_stat_map

    def save_site_coverage_stats(self, filename):
        n_models = 0
        self.finalize_stats()
        if not self.low_memory:
            for model_name in self.rank_dicts.keys():
                self.compute_site_coverage_stats(model_name=model_name)
                n_models += 1
        else:
            n_models = len(self.site_id_stat_maps.keys())
        if n_models > 0:
            self.logger.info(f"Computed coverage stats for {n_models} models; saving to pickle.")
            os.makedirs(self.config.coverage_stats_dir, exist_ok=True)
            coverage_filepath = os.path.join(self.config.coverage_stats_dir, filename)
            with open(coverage_filepath, 'wb') as outfile:
                pickle.dump(self.site_id_stat_maps, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.logger.info(f"No models to compute coverage stats for; saving nothing.")


if __name__ == "__main__":
    import doctest
    doctest.testmod()