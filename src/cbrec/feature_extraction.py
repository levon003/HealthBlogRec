"""
This module is for generating features from user/site pair (USP) triples.
A USP triple consists of a source, a target, and an alternative USP.
Each USP has a fixed-length feature representation.

Note that some features need to be computed as crosses or differences between features (e.g. "same health condition?").
Not sure yet how to handle this, but marking such features as Feature Crosses (FCs) in the list below.

User/site metadata:
- Site health condition?
- User health condition?

Network features:
- Indegree
- Outdegree
- Weakly connected component size
- (FC) Same weakly connected component?
- (FC) Is reciprocal?
- (FC) Is friend-of-friend? (Triadic closure)

Activity features:
- Recent journals
- Recent amps
- Recent comments
- Recent guestbooks
- Time to most recent journal (times recent journals > 1)
- Time to most recent amp
- Time to most recent comment
- Time to most recent guestbook
- Time to first journal update (on this site)

Journal update features:
- roBERTa embedding of three most recent journal updates
- Author role (max, mid, and min of most recent updates)
"""

from datetime import datetime
import multiprocessing as mp
from collections import OrderedDict
import subprocess as sp
import numpy as np
import logging
import queue
import json
import os

from . import featuredb


POISON = "StopProcessing"
FORCE_COMMIT = "ForceCommit"


class FeatureGenerator:
    def __init__(self, config, graph, activity_manager):
        self.config = config
        self.logger = logging.getLogger('cbrec.feature_generation.FeatureGenerator')
        
        self.graph = graph
        self.activity_manager = activity_manager

        # note: potential danger with caching features. If the activity manager or graph change between calls to generate_*_features(), then using the cache value may be incorrect
        # the solution is to make sure you don't do that, which should be easy given the unique timestamp
        # note 2: this cache basically doesn't do anything, since it replies on repeat requests for the same USP at the same interaction_timestamp, which should basically not happen
        self.feature_arr_cache = LruCache(maxsize=10) 
    
    def generate_user_features(self, usp, interaction_timestamp, use_cache=False):
        if use_cache:
            arg_hash = (int(usp[0]), int(usp[1]), interaction_timestamp)
            if arg_hash in self.feature_arr_cache:
                #self.logger.debug("User feature cache hit.")
                return self.feature_arr_cache[arg_hash]

        user_id, site_id = usp
        feat_arr = np.empty(self.config.user_feature_count, dtype=featuredb.NUMPY_DTYPE)
        
        # network features
        indegree = self.graph.get_indegree(user_id)
        outdegree = self.graph.get_outdegree(user_id)
        component_size = self.graph.get_component_size(user_id)
        feat_arr[0] = indegree
        feat_arr[1] = outdegree
        feat_arr[2] = component_size
        
        # activity features
        start_offset = 3
        for i, key in enumerate(['journal_user', 'amp', 'comment', 'guestbook']):
            rac = self.activity_manager.get_activity_counter(key)
            n_recent = rac.get_count(user_id)
            if n_recent > 0:
                most_recent = rac.get_most_recent_activity(user_id)
                time_to_most_recent = interaction_timestamp - most_recent
                time_to_most_recent = max(time_to_most_recent, 0)  # this would
                # convert difference from ms to hours
                time_to_most_recent /= (1000 * 60 * 60)
            else:
                # no recent interactions, so need to choose a default for time_to_most_recent
                # options for a time_to_most_recent:
                # 1. rac.activity_count_duration_ms / (1000 * 60 * 60) : the maximum elapsed time
                # 2. -1 : a special value that should be handled differently by the model
                # 3. 0 : a special value that should be handled differently by the model, but may offer more reasonable default behavior (due to quirks of multiplying by zero)
                time_to_most_recent = 0
            feat_arr[start_offset + 2*i] = n_recent
            feat_arr[start_offset + 2*i + 1] = time_to_most_recent
        first_update_timestamp = self.activity_manager.get_first_journal_update_timestamp(user_id)
        time_to_first_update = interaction_timestamp - first_update_timestamp
        time_to_first_update = max(time_to_first_update, 0)
        time_to_first_update /= (1000 * 60 * 60)  # convert to hours
        feat_arr[11] = time_to_first_update
        
        if use_cache:
            self.feature_arr_cache[arg_hash] = feat_arr
        return feat_arr
    
    def generate_user_pair_features(self, usp1, usp2, interaction_timestamp):
        """
        Note: We don't cache user pair features, since they're very unlikely to be reusable.
        """

        user_id1, site_id1 = usp1
        user_id2, site_id2 = usp2
        feat_arr = np.empty(self.config.user_pair_feature_count, dtype=featuredb.NUMPY_DTYPE)
        
        # network features
        are_weakly_connected = self.graph.are_weakly_connected(user_id1, user_id2)
        is_fof = self.graph.compute_is_friend_of_friend(user_id1, user_id2)
        is_reciprocal = self.graph.is_reciprocal(user_id1, user_id2)
        feat_arr[0] = are_weakly_connected
        feat_arr[1] = is_fof
        feat_arr[2] = is_reciprocal

        return feat_arr
        
        
class DatabaseWriter:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('cbrec.feature_extraction.DatabaseWriter')

        self.feature_id_counter = -1
        self.feature_arr_cache = LruCache(maxsize=config.feature_arr_cache_size)

        self.__instantiate_manager()

    def __getstate__(self):
        # Copy the object's state
        state = self.__dict__.copy()
        # Remove unpicklable entries
        del state['db_queue']
        del state['writer_process']

        # To avoid losing data that's stuck in the queue, we process all currently queued data
        if self.config.use_writer_process and self.writer_process is not None:
            self.logger.info(f"Waiting on a queue backlog of approximately {self.db_queue.qsize()} items during state export.")
            self.db_queue.put(FORCE_COMMIT)
            self.db_queue.join()
            assert self.writer_process.uncommitted_insert_count == 0
            # Do note that the writer process needs to be properly stopped before reinstantiation, if for no other reason than to give up the file handles

        return state


    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        # Re-initialize managed queue
        self.__instantiate_manager()


    def __instantiate_manager(self):
        manager = mp.Manager()
        self.db_queue = manager.Queue()
        self.writer_process = None


    def set_config(self, config):
        self.config = config


    def start_process(self):
        if self.config.use_writer_process:
            self.set_feature_id_counter()
            self.writer_process = WriterProcess(self.config, self.db_queue)
            self.writer_process.start()
            self.logger.info(f"Started DatabaseWriter's WriterProcess.")
        else:
            self.logger.warning(f"Told to start writer process, but use_writer_process is false.")


    def stop_process(self, wait=True):
        if not self.config.use_writer_process:
            self.logger.warning(f"Told to stop writer process, but use_writer_process is false.")
        if self.writer_process is not None:
            self.db_queue.put(POISON)
            self.logger.info(f"Inserting shutdown poison into queue with a backlog of approximately {self.db_queue.qsize()} items.")
            if wait:
                self.writer_process.join()
                self.writer_process = None
        else:
            self.logger.info("Ignoring request to stop WriterProcess; not running.")
    

    def set_feature_id_counter(self):
        """
        feature_id should be globally unique, so check for existing feature_id entries in the database.

        Note: does a database read, so potential issues with calling this while the WriterProcess is running.
        """
        db = featuredb.get_db_by_filepath(self.config.feature_db_filepath)
        with db:
            featuredb.create_db(db)
            max_feature_id = featuredb.get_max_feature_id(db)
            if max_feature_id != self.feature_id_counter:
                self.logger.info(f"Updating current feature_id counter from {self.feature_id_counter} to {max_feature_id} based on existing database entries.")
                self.feature_id_counter = max_feature_id

    
    def get_current_metadata_id(self):
        """
        metadata_id should be globally unique, so check for existing metadata_id entries in the database.

        Note: does a database read, so potential issues with calling this while the WriterProcess is running.
        """
        db = featuredb.get_db_by_filepath(self.config.feature_db_filepath)
        with db:
            featuredb.create_db(db)
            max_metadata_id = featuredb.get_max_metadata_id(db)
            return max_metadata_id


    def save_feature_array(self, ndarray):
        """
        Saves a feature array to the database.

        If this array already exists in the cache, instead return the cached feature_id.
        """
        array_hash = ndarray.data.tobytes()
        if array_hash in self.feature_arr_cache:
            #self.logger.debug("Cache hit.")
            return self.feature_arr_cache[array_hash]
        else:
            # an array with these bytes is not already cached
            self.feature_id_counter += 1
            feature_id = self.feature_id_counter
            self.feature_arr_cache[array_hash] = feature_id  # cache the array and its associated feature_id
            self.db_queue.put(('feature', feature_id, ndarray))
            return feature_id

    def save_rec_context(self, rec_context):
        if rec_context.is_initiation_eligible:
            if rec_context.is_test_period:
                self.save_test_context(rec_context)
            else:
                self.save_triples(rec_context)
        self.save_metadata(rec_context)

    def save_metadata(self, rec_context):
        self.db_queue.put(('metadata',
            rec_context.md
        ))

    def save_triples(self, rec_context):
        """
        Generate and save features for this triple.
        """
        metadata_id = rec_context.metadata_id
        interaction_timestamp = rec_context.timestamp

        for triple in rec_context.triples:
            source_usp, target_usp, alt_usp, source_feat_arr, target_feat_arr, alt_feat_arr, source_target_feat_arr, source_alt_feat_arr = triple

            source_feature_id = self.save_feature_array(source_feat_arr)
            target_feature_id = self.save_feature_array(target_feat_arr)
            alt_feature_id = self.save_feature_array(alt_feat_arr)
            source_target_shared_feature_id = self.save_feature_array(source_target_feat_arr)
            source_alt_shared_feature_id = self.save_feature_array(source_alt_feat_arr)
        
            self.db_queue.put(('triple', 
                interaction_timestamp,
                metadata_id, 
                source_usp, source_feature_id,
                target_usp, target_feature_id,
                alt_usp, alt_feature_id,
                source_target_shared_feature_id, source_alt_shared_feature_id))

    def save_test_context(self, test_context):
        metadata_id = test_context.metadata_id
        source_usp_arr_id = self.save_feature_array(test_context.source_usp_arr)#.astype(featuredb.NUMPY_DTYPE, casting='unsafe'))
        candidate_usp_arr_id = self.save_feature_array(test_context.candidate_usp_arr)#.astype(featuredb.NUMPY_DTYPE, casting='unsafe'))
        target_inds_id = self.save_feature_array(test_context.target_inds)#.astype(featuredb.NUMPY_DTYPE, casting='unsafe'))
        source_usp_mat_id = self.save_feature_array(test_context.source_usp_mat)
        candidate_usp_mat_id = self.save_feature_array(test_context.candidate_usp_mat)
        user_pair_mat_id = self.save_feature_array(test_context.user_pair_mat)
        self.db_queue.put(('test_context',
            metadata_id,
            source_usp_arr_id,
            candidate_usp_arr_id,
            target_inds_id,
            source_usp_mat_id,
            candidate_usp_mat_id,
            user_pair_mat_id,
        ))


class WriterProcess(mp.Process):
    def __init__(self, config, queue, **kwargs):
        super(WriterProcess, self).__init__()
        self.logger = logging.getLogger('cbrec.feature_extraction.WriterProcess')
        self.config = config
        self.queue = queue
        self.kwargs = kwargs

        self.uncommitted_insert_count = 0

    def create_db_table(self):
        db = featuredb.get_db_by_filepath(self.config.feature_db_filepath)
        try:
            featuredb.create_db(db)
            self.logger.info(f"Instantiated feature db at path '{self.config.feature_db_filepath}'.")
        finally:
            db.close()

    def summarize_metadata_type(self, md, offset=0):
        """
        Debugging function for checking types in the metadata dict.
        """
        if type(md) == dict:
            return_str = f"dict with {len(md)} keys\n"
            for key, value in md.items():
                assert type(key) == str
                return_str += f"{'  ' * offset}{key}: {self.summarize_metadata_type(value, offset=offset+1)}\n"
            return return_str
        elif type(md) == list:
            if len(md) > 0:
                return f"list of {len(md)} {self.summarize_metadata_type(md[0], offset=offset+1)}"
            else:
                return "empty list"
        elif type(md) == set:
            return f"set of size {len(md)}"
        else:
            return str(type(md))

    def report_item_counts(self, item_type_counts, db, get_full_counts=True):
        full_item_type_counts = {}
        if get_full_counts:
            # get FULL counts from the database and the metadata file, as appropriate
            res = sp.run(["wc","-l",self.config.metadata_filepath], stdout=sp.PIPE)
            if res.returncode == 0:
                count = int(res.stdout.decode().split()[0])
                full_item_type_counts['metadata'] = count
            
            for item_type in ['feature', 'triple', 'test_context']:
                res = db.execute("SELECT COUNT(*) AS count FROM " + item_type).fetchone()
                count = res['count']
                full_item_type_counts[item_type] = count

        self.logger.info("DatabaseWriter summary: ")
        for item_type, count in item_type_counts.items():
            self.logger.info(f"    Wrote {count} items of type '{item_type}'{' (found ' + str(full_item_type_counts[item_type]) + ' total)' if item_type in full_item_type_counts else ''}.")

    def register_db_task(self, db, force_commit=False):
        """
        Used to defer commits until "necessary".
        """
        if self.uncommitted_insert_count > 0 and (force_commit or self.uncommitted_insert_count >= self.config.feature_db_max_commit_size or self.queue.qsize() == 0):
            db.commit()
            for i in range(self.uncommitted_insert_count):
                self.queue.task_done()
            if force_commit:
                self.logger.debug(f"Force committed {self.uncommitted_insert_count} inserts.")
            self.uncommitted_insert_count = 0

    def run(self):
        item_type_counts = {'feature': 0, 'triple': 0, 'metadata': 0, 'test_context': 0}
        # if appropriate, delete existing
        if self.config.should_clear_feature_db_filepath:
            if os.path.exists(self.config.feature_db_filepath):
                os.remove(self.config.feature_db_filepath)
                self.logger.info("Deleted existing feature db.")
            else:
                self.logger.info("No existing feature db file to delete.")
        if self.config.should_clear_metadata_filepath:
            if os.path.exists(self.config.metadata_filepath):
                os.remove(self.config.metadata_filepath)
                self.logger.info("Deleted existing metadata.")
            else:
                self.logger.info("No existing metadata file to delete.")
        # TODO Given a metadata_id, consider clearing away some of the lines in the file and entries in the database
        # or maybe splitting the metadata file and clearing the database??  or setting the metadata_id based on the database and propagating back to the rdg...

        # write to metadata_filepath in append mode by default
        with open(self.config.metadata_filepath, 'a') as metadata_file:
            self.create_db_table()
            db = featuredb.get_db_by_filepath(self.config.feature_db_filepath)
            self.logger.info("Opened metadata and feature file descriptors; listening for inputs.")
            try:
                while True:
                    # create (and remove existing) table to be inserted into
                    item = self.queue.get()
                    if item == POISON:
                        self.logger.info("Terminating WriterProcess.")
                        self.queue.task_done()
                        break
                    elif item == FORCE_COMMIT:
                        self.logger.info(f"Received force commit request; currently {self.uncommitted_insert_count} uncommitted inserts.")
                        self.register_db_task(db, force_commit=True)
                        self.queue.task_done()
                        continue
                    item_type = item[0]
                    #self.logger.info(f"Pulled '{item_type}' item from queue. Backlog size: {self.queue.qsize()}")
                    if item_type == 'feature':
                        featuredb.insert_feature(db, *item[1:])
                        self.uncommitted_insert_count += 1
                        self.register_db_task(db)
                    elif item_type == 'triple':
                        featuredb.insert_triple(db, *item[1:])
                        self.uncommitted_insert_count += 1
                        self.register_db_task(db)
                    elif item_type == 'metadata':
                        metadata_dict = item[1]
                        try:
                            metadata_file.write(json.dumps(metadata_dict) + "\n")
                        except:
                            self.logger.info(self.summarize_metadata_type(metadata_dict))
                        self.queue.task_done()
                    elif item_type == 'test_context':
                        featuredb.insert_test_context(db, *item[1:])
                        self.uncommitted_insert_count += 1
                        self.register_db_task(db)
                    else:
                        raise ValueError("Unknown item type.")
                    item_type_counts[item_type] += 1
                self.register_db_task(db, force_commit=True)
                assert self.uncommitted_insert_count == 0
                self.report_item_counts(item_type_counts, db, get_full_counts=False)
            except Exception as ex:
                self.logger.error(ex)
                raise ex
            finally:
                db.close()
        self.logger.info("File descriptors closed; terminating WriterProcess.")


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
            
    def clear_cache(self):
        for key in list(self.keys()):
            del self[key]
