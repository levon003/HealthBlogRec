
import os
import logging
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import cbrec.data
import cbrec.text.embeddingdb
import cbrec.text.journalid


class TextLoader:
    """
    TextLoader provides functions to cache texts.
    
    By default, text loader caches the entirety of the text embeddings in-memory.
    
    Also maintains a cache of pooled embeddings.
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("cbrec.modeling.text_loader.TextLoader")
        
        dm = cbrec.data.DataManager(config, load_ints=False, load_journals=True)
        self.journal_id_lookup = cbrec.text.journalid.JournalIdLookup(config, dm)
        
        self.journal_embedding_map = {}
        
        self.feature_cache = LruCache(maxsize=50000)
        
        
    def cache_all_journals(self, use_test_embeddings=True, verify_journal_metadata=True):
        if use_test_embeddings:
            self.config.text_feature_db_filepath = os.path.join(self.config.feature_data_dir, 'test_text_feature.sqlite')
        db = cbrec.text.embeddingdb.get_text_feature_db(self.config)
        with db:
            n_missing_journals = 0
            n_journals = 0
            for text in tqdm(cbrec.text.embeddingdb.stream_text_features(db), total=998905):
                journal_id = text['text_id']
                n_journals += 1
                if verify_journal_metadata and journal_id not in self.journal_id_lookup.journal_id_set:
                    n_missing_journals += 1
                    continue
                self.journal_embedding_map[journal_id] = text['feature_arr']
            if n_missing_journals > 0:
                self.logger.warn(f"{n_missing_journals} / {n_journals} missing from journal_df.")
        self.logger.info(f"Cached all {len(self.journal_embedding_map)} journal embeddings.")
        
    def get_journal_embedding(self, journal_id):
        return self.journal_embedding_map[journal_id]
    
    def get_journal_embeddings(self, usp, timestamp):
        journal_ids = self.journal_id_lookup.get_journal_updates_before(usp, timestamp)
        if len(journal_ids) < 3:
            raise ValueError(f"Insufficient texts: {usp}@{timestamp} n={len(journal_ids)}")
        try:
            return [self.get_journal_embedding(journal_id) for journal_id in journal_ids]
        except KeyError:
            raise KeyError(f"USP {usp} at timestamp {timestamp} returned journal ids ({journal_ids}) that weren't present in the cache.")
            
    def get_text_features(self, usp, timestamp):
        """
        This is the method that should be used, since it adds a caching layer.
        """
        if (usp, timestamp) in self.feature_cache:
            return self.feature_cache[(usp, timestamp)]
        text_arrs = self.get_journal_embeddings(usp, timestamp)
        if (self.config.pool_text_feature == 0):
            text_arr = np.mean(text_arrs, axis=0)  # mean pool the available texts
        elif (self.config.pool_text_feature == 1):
            text_arr = np.concatenate([text_arrs[0], text_arrs[1], text_arrs[2]])
        elif (self.config.pool_text_feature == 2):
            text_arr = np.amax(text_arrs, axis=0)
        self.feature_cache[(usp, timestamp)] = text_arr
        return text_arr
        
        
class LruCache(OrderedDict):
    """
    An OrderedDict with limited size, evicting the least recently looked-up key when full.

    Note that a contains check (key in self) does not count as a look-up.

    See source: https://docs.python.org/3/library/collections.html#ordereddict-examples-and-recipes
    """

    def __init__(self, maxsize=2048, *args, **kwds):
        self.maxsize = maxsize
        super().__init__(*args, **kwds)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]
