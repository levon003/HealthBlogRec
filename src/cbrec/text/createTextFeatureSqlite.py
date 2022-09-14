#!/usr/bin/env python3
# Creates a table that contains feature data

import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

import sys
import os
from tqdm import tqdm
import itertools
from datetime import datetime
import sqlite3
import multiprocessing as mp
import threading
import argparse
import numpy as np

# HuggingFace packages
import transformers
import tokenizers
import torch

try:
    import cbrec
except:
    sys.path.append("/home/lana/levon003/repos/recsys-peer-match/src")

from cbrec import featuredb
from cbrec import genconfig
from cbrec.text import textdb


POISON = "POISON"  # an object that is recognized as a "stop processing" signal by the writer process

N_PROCESSES = 32  # number of extractor processes to use
# in the case of this script, that's the number of roBERTa models that will be loaded and used in-memory


def create_table(db, drop_table=False):
    if drop_table:
        db.execute("DROP TABLE IF EXISTS textFeature")
    create_table_command = """
    CREATE TABLE IF NOT EXISTS textFeature (
          text_id TEXT PRIMARY KEY,
          feature_arr NDARRAY NOT NULL
        )
    """
    db.execute(create_table_command)
    db.commit()


class ExtractorProcess(mp.Process):
    """
    ExtractorProcess receives objects from an input_queue, processes them, and writes the extracted result to an output_queue.
    
    """
    def __init__(self, keep_alive, input_queue, output_queue, ready_barrier, **kwargs):
        """
        :keep_alive - mp.Value that indicates if queue processing should continue
        :ready_barrier - Barrier to wait on before queue processing will begin.  Enables one to sync multiple extractors.
        """
        super(ExtractorProcess, self).__init__()
        self.keep_alive = keep_alive
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.ready_barrier = ready_barrier
        self.kwargs = kwargs

    def run(self):
        roberta_tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
        roberta_model = transformers.RobertaModel.from_pretrained('roberta-base')
        roberta_model.eval()
        torch.set_num_threads(4)
        self.ready_barrier.wait()

        logging.info(f"Extractor process started queue processing at {str(datetime.now())}.")
        while True:
            result = self.input_queue.get()
            if result == POISON:
                break
            text_id, text = result
            tokenized = roberta_tokenizer(text, 
                            padding=False,
                            truncation=True, 
                            return_tensors="pt")
            input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']

            with torch.no_grad():
                outputs = roberta_model(input_ids, attention_mask=attention_mask)
                lhs = outputs['last_hidden_state'].numpy()
            last_hidden_state = lhs[0,:,:]
            mean_pool = np.mean(last_hidden_state, axis=0)
            text_feature_arr = mean_pool.astype(featuredb.NUMPY_DTYPE)
            output_tuple = (text_id, text_feature_arr)
            self.output_queue.put(output_tuple)
            self.input_queue.task_done()
        if self.keep_alive.value:
            logging.error(f"Terminating queue processing early (keep-alive={self.keep_alive.value}).")
        else:
            logging.info(f"Finished queue processing (keep-alive={self.keep_alive.value}).")

    
class WriterProcess(mp.Process):
    """
    WriterProcess writes received data to the database.
    
    It is instantiated with a queue (that it reads from) and a filename (that it writes to).
    """
    def __init__(self, queue, config, **kwargs):
        super(WriterProcess, self).__init__()
        self.queue = queue
        self.config = config
        self.kwargs = kwargs

    def run(self):
        db = featuredb.get_text_feature_db(self.config)
        try:
            # note: assume relevant table already exists
            #db.isolation_level = None  # Set to autocommit mode; control transactions using explicit begins
            #db.execute("BEGIN TRANSACTION")
            processed_count = 0  # tracks the number of items processed so far
            s = datetime.now()
            logging.info(f"Insertion process started queue processing at {str(s)}.")
            while True:
                result = self.queue.get()
                if result == POISON:
                    break
                text_id, text_feature_arr = result
                db.execute(
                    'INSERT INTO textFeature (text_id, feature_arr) VALUES (?, ?)',
                    (text_id, text_feature_arr)
                )

                processed_count += 1
                if processed_count % 100000 == 0:
                    db.commit()
                    logging.info(f"Rows committed after {datetime.now() - s}. ({processed_count} total)")
            db.commit()
            logging.info(f"Final rows committed after {datetime.now() - s}. ({processed_count} total)")
        except mp.TimeoutError:
            logging.error("Timeout waiting for a process.\n")
        finally:
            db.close()
        logging.info("Db closed, insertion process terminating.")


def process_ids_to_features(config, text_ids, id_limit=None, should_check_for_duplicates=True, max_results=10000, n_processes=N_PROCESSES, dryrun=False):
    db = featuredb.get_text_feature_db(config)
    try:
        create_table(db)
        
        if should_check_for_duplicates:
            total_existing = 0
            total_duplicates = 0
            result = db.execute("SELECT text_id FROM textFeature")
            for row in result:
                total_existing += 1
                existing_text_id = row['text_id']
                if existing_text_id in text_ids:
                    text_ids.remove(existing_text_id)
                    total_duplicates += 1
            logging.info(f"After duplicate check, removed {total_duplicates} duplicate ids from {total_existing} total existing ids.")
        else:
            logging.info("Table created; not checking for duplicates.")

    finally:
        db.close()
        
    if dryrun:
        logging.info("Dryrun; terminating without starting processes.")
        return
    
    with mp.Manager() as manager:        
        td = textdb.TextDatabase(config)
        text_db = td.get_text_db()
        try:
            results = []
            keep_extractors_alive = manager.Value(bool, value=True)
            text_queue = manager.Queue(maxsize=max_results)
            result_queue = manager.Queue()
            extractor_ready_barrier = manager.Barrier(n_processes + 1)
            writer_process = WriterProcess(queue=result_queue, config=config)
            writer_process.start()
            worker_processes = []
            for i in range(n_processes):
                worker_process = ExtractorProcess(keep_alive=keep_extractors_alive, input_queue=text_queue, output_queue=result_queue, ready_barrier=extractor_ready_barrier)
                worker_process.daemon = True
                worker_process.start()
                worker_processes.append(worker_process)
            try:
                extractor_ready_barrier.wait(timeout=120)
            except threading.BrokenBarrierError:
                logging.error("One or more of the extractor processes failed to initialize.")
                return
            
            logging.info("Processes started.")
            processed_count = 0
            for i, text_id in tqdm(enumerate(text_ids), total=len(text_ids), desc="Retrieving texts"):
                # get the text from a feature db
                raw_title, raw_body = td.get_raw_journal_text_from_db(text_db, text_id)
                text = textdb.clean_journal(raw_title, raw_body)
                text_queue.put((text_id, text))
                processed_count += 1
                if id_limit is not None and i > id_limit:
                    # done processing ids, break out of the async loop
                    break
            logging.info(f"Queued all texts; ~{text_queue.qsize()} items still in text queue (and ~{result_queue.qsize()} items waiting in the result queue).")
            text_queue.join()
            # all text tasks processed, stop the extractor processes
            s = datetime.now()
            keep_extractors_alive.value = False
            for worker_process in worker_processes:
                text_queue.put(POISON)
            for worker_process in worker_processes:
                worker_process.join()
            logging.info(f"Signaled and ended worker processes in {datetime.now() - s}.")
            logging.info(f"Finished processing {processed_count} texts (approximately {result_queue.qsize()} items still in result queue).")

            # Stop the writer process, ensuring that the queue has been fully processed
            result_queue.put(POISON)
            writer_process.join()
            
            logging.info("Terminated writer process.")
        finally:
            text_db.close()
            logging.info("Text database connection closed.")
    logging.info("Finished.")

    
def get_text_ids(text_id_filepath):
    text_ids = set()
    with open(text_id_filepath, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line != '':
                text_ids.add(line)
    logging.info(f"Identified {len(text_ids)} text ids.")
    return text_ids


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--raw-dir', dest='raw_dir', default='/home/zlevonian/follows/raw')
    parser.add_argument('--text-id-txt', dest='text_id_filepath', required=True)
    parser.add_argument('--text-feature-db-filename', dest='text_feature_db_filename', required=False, default="")
    parser.add_argument('--dryrun', dest='dryrun', required=False, action="store_true", default=False)
    parser.add_argument('--n-processes', dest='n_processes', required=False, default=N_PROCESSES)
    args = parser.parse_args()
    
    text_ids = get_text_ids(args.text_id_filepath)
    
    config = genconfig.Config()
    
    text_feature_db_filename = args.text_feature_db_filename
    if text_feature_db_filename.strip() != "":
        text_feature_db_filename = text_feature_db_filename.strip()
        if not text_feature_db_filename.endswith('.sqlite'):
            raise ValueError(f"Using '{text_feature_db_filename}', which doesn't end with .sqlite. Confirm path?")
        config.text_feature_db_filepath = os.path.join(config.feature_data_dir, text_feature_db_filename)
        logging.info(f"Manual output path override: Writing to text_feature_db '{config.text_feature_db_filepath}'.")
    
    process_ids_to_features(config, text_ids, id_limit=None, n_processes=int(args.n_processes), dryrun=args.dryrun)
    

if __name__ == "__main__":
    main()
