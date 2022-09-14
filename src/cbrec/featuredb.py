"""

The feature database has multiple entities and needs, disparately for training and testing.
feature_arrays
initiation_metadata
train_triples
test_execution

For training, we need to generate triples.

"""


import numpy as np
import sqlite3
import os
import logging
import json

from . import paths
from . import genconfig

"""
The dtype to use in feature DB array storage.

Not in the config because it is a static external setting.
"""
NUMPY_DTYPE = np.float32

# Note: We define the integer number (that must be <= 255) manually, but we could use the dtype.num defined for each of the numpy types, if we wanted to support every type
DTYPE_BYTE_MAP = {
    np.dtype(np.float32): 0,
    np.dtype(np.int64): 1,
}
BYTE_DTYPE_MAP = {value: key for key, value in DTYPE_BYTE_MAP.items()}


def adapt_numpy_ndarray(arr):
    # see discussion: https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database/18622264
    if arr.dtype not in DTYPE_BYTE_MAP:
        raise ValueError(f"Expected dtype {NUMPY_DTYPE}, got {arr.dtype} instead.")
    else:
        arr_bytes = arr.tobytes()
        dtype_byte = bytes([DTYPE_BYTE_MAP[arr.dtype],])
        combined_bytes = dtype_byte + arr_bytes
        assert len(combined_bytes) == len(dtype_byte) + len(arr_bytes)
        return combined_bytes


def convert_numpy_ndarray(text):
    dtype_byte = text[0]
    assert dtype_byte in BYTE_DTYPE_MAP
    dtype = BYTE_DTYPE_MAP[dtype_byte]
    arr_bytes = text[1:]
    return np.frombuffer(arr_bytes, dtype=dtype)


def adapt_json_dict(python_dict):
    return json.dumps(python_dict)


def convert_json_dict(text):
    return json.loads(text)


def get_feature_db(config):
    #feature_db_filepath = os.path.join(config.feature_data_dir, 'feature.sqlite')
    #db = get_db_by_filepath(feature_db_filepath)
    db = get_db_by_filepath(config.feature_db_filepath)
    return db


def get_text_feature_db(config):
    db = get_db_by_filepath(config.text_feature_db_filepath)
    return db


def get_db_by_filepath(db_filepath):
    sqlite3.register_adapter(np.ndarray, adapt_numpy_ndarray)
    sqlite3.register_converter("NDARRAY", convert_numpy_ndarray)
    #sqlite3.register_adapter(dict, adapt_json_dict)
    #sqlite3.register_converter("JSON", convert_json_dict)
    db = sqlite3.connect(
        db_filepath,
        detect_types=sqlite3.PARSE_DECLTYPES
    )
    db.row_factory = sqlite3.Row
    return db


def create_db(db):
    create_table_command = """
    CREATE TABLE IF NOT EXISTS feature (
          feature_id INTEGER PRIMARY KEY,
          feature_arr NDARRAY NOT NULL
        )
    """
    db.execute(create_table_command)
    create_table_command = """
    CREATE TABLE IF NOT EXISTS test_context (
            metadata_id INTEGER PRIMARY KEY,
            source_usp_arr_id INTEGER NOT NULL,
            candidate_usp_arr_id INTEGER NOT NULL,
            target_inds_id INTEGER NOT NULL,
            source_usp_mat_id INTEGER NOT NULL,  /* dim: len(source_usp_arr) X n_features */
            candidate_usp_mat_id INTEGER NOT NULL,  /* dim: len(candidate_usp_arr) X n_features */
            user_pair_mat_id INTEGER NOT NULL,  /* dim: (len(source_usp_arr) X len(candidate_usp_arr)) X n_features */
            FOREIGN KEY(source_usp_arr_id) REFERENCES feature(feature_id),
            FOREIGN KEY(candidate_usp_arr_id) REFERENCES feature(feature_id),
            FOREIGN KEY(target_inds_id) REFERENCES feature(feature_id),
            FOREIGN KEY(source_usp_mat_id) REFERENCES feature(feature_id),
            FOREIGN KEY(candidate_usp_mat_id) REFERENCES feature(feature_id),
            FOREIGN KEY(user_pair_mat_id) REFERENCES feature(feature_id)
        )
    """
    db.execute(create_table_command)
    create_table_command = """
    CREATE TABLE IF NOT EXISTS triple (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          interaction_timestamp INTEGER NOT NULL,
          metadata_id INTEGER NOT NULL,
          source_user_id INTEGER NOT NULL,
          source_site_id INTEGER NOT NULL,
          source_feature_id INTEGER NOT NULL,
          target_user_id INTEGER NOT NULL,
          target_site_id INTEGER NOT NULL,
          target_feature_id INTEGER NOT NULL,
          alt_user_id INTEGER NOT NULL,
          alt_site_id INTEGER NOT NULL,
          alt_feature_id INTEGER NOT NULL,
          source_target_shared_feature_id INTEGER,
          source_alt_shared_feature_id INTEGER,
          FOREIGN KEY(source_feature_id) REFERENCES feature(feature_id),
          FOREIGN KEY(target_feature_id) REFERENCES feature(feature_id),
          FOREIGN KEY(alt_feature_id) REFERENCES feature(feature_id),
          FOREIGN KEY(source_target_shared_feature_id) REFERENCES feature(feature_id),
          FOREIGN KEY(source_alt_shared_feature_id) REFERENCES feature(feature_id)
        )
    """
    db.execute(create_table_command)
    db.commit()
    

def get_max_feature_id(db):
    cursor = db.execute("SELECT MAX(feature_id) AS max_feature_id FROM feature")
    result = cursor.fetchone()
    if result is None:
        return -1
    else:
        max_feature_id = result['max_feature_id']
        return max_feature_id if max_feature_id is not None else -1


def get_max_metadata_id(db):
    cursor = db.execute("SELECT MAX(metadata_id) AS max_metadata_id FROM test_context")
    result = cursor.fetchone()
    if result is None or result['max_metadata_id'] is None:
        cursor = db.execute("SELECT MAX(metadata_id) AS max_metadata_id FROM triple")
        result = cursor.fetchone()
        if result is None or result['max_metadata_id'] is None:
            return -1
        else:
            return result['max_metadata_id']
    else:
        max_metadata_id = result['max_metadata_id']
        return max_metadata_id if max_metadata_id is not None else -1


def insert_feature(db, feature_id, feature_arr):
    db.execute("""
        INSERT INTO feature (feature_id, feature_arr)
        VALUES (?, ?)
    """, (feature_id, feature_arr))
    

def insert_triple(db, 
                interaction_timestamp,
                metadata_id, 
                source_usp, source_feature_id,
                target_usp, target_feature_id,
                alt_usp, alt_feature_id,
                source_target_shared_feature_id, source_alt_shared_feature_id
                 ):
    db.execute("""
            INSERT INTO triple (interaction_timestamp,
                metadata_id, 
                source_user_id, source_site_id, source_feature_id,
                target_user_id, target_site_id, target_feature_id,
                alt_user_id, alt_site_id, alt_feature_id,
                source_target_shared_feature_id, source_alt_shared_feature_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (interaction_timestamp, metadata_id, 
            int(source_usp[0]), int(source_usp[1]), source_feature_id,
            int(target_usp[0]), int(target_usp[1]), target_feature_id,
            int(alt_usp[0]), int(alt_usp[1]), alt_feature_id,
            source_target_shared_feature_id, source_alt_shared_feature_id))


def insert_test_context(db, 
    metadata_id,
    source_usp_arr_id, candidate_usp_arr_id, target_inds_id, 
    source_usp_mat_id, candidate_usp_mat_id, user_pair_mat_id):
    db.execute("""
            INSERT INTO test_context (metadata_id,
    source_usp_arr_id, candidate_usp_arr_id, target_inds_id, 
    source_usp_mat_id, candidate_usp_mat_id, user_pair_mat_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (metadata_id,
    source_usp_arr_id, candidate_usp_arr_id, target_inds_id, 
    source_usp_mat_id, candidate_usp_mat_id, user_pair_mat_id))


def get_feature(db, feature_id):
    cursor = db.execute('SELECT feature_arr FROM feature WHERE feature_id = ?', (feature_id,))
    if cursor is None:
        return ValueError(f"Failed execution while retrieving features for id {feature_id}.")
    rows = cursor.fetchall()
    if len(rows) == 0:
        return ValueError(f"No feature found with id {feature_id}.")
    row = rows[0]
    return row['feature_arr']


def stream_triples(db, limit=None):
    command = """
    SELECT 
        interaction_timestamp, 
        metadata_id,
        source_user_id,
        source_site_id,
        target_user_id,
        target_site_id,
        alt_user_id,
        alt_site_id,
        f1.feature_arr as source_feature_arr, 
        f2.feature_arr as target_feature_arr, 
        f3.feature_arr as alt_feature_arr,
        f4.feature_arr as source_target_feature_arr,
        f5.feature_arr as source_alt_feature_arr
    FROM triple
    LEFT JOIN feature AS f1 ON source_feature_id = f1.feature_id
    LEFT JOIN feature AS f2 ON target_feature_id = f2.feature_id
    LEFT JOIN feature AS f3 ON alt_feature_id = f3.feature_id
    LEFT JOIN feature AS f4 ON source_target_shared_feature_id = f4.feature_id
    LEFT JOIN feature AS f5 ON source_alt_shared_feature_id = f5.feature_id
    ORDER BY RANDOM()
    """
    if limit is not None:
        command += f"\nLIMIT {limit}"
    cursor = db.execute(command)
    if cursor is None:
        return ValueError("Null cursor.")
    row = cursor.fetchone()
    while row is not None:
        yield row
        row = cursor.fetchone()
        
def stream_test_contexts(db, config):
    cursor = db.execute("""
    SELECT 
        metadata_id, 
        f1.feature_arr as source_usp_arr,
        f2.feature_arr as candidate_usp_arr,
        f3.feature_arr as target_inds,
        f4.feature_arr as source_usp_mat,
        f5.feature_arr as candidate_usp_mat,
        f6.feature_arr as user_pair_mat
    FROM test_context
    LEFT JOIN feature AS f1 ON source_usp_arr_id = f1.feature_id
    LEFT JOIN feature AS f2 ON candidate_usp_arr_id = f2.feature_id
    LEFT JOIN feature AS f3 ON target_inds_id = f3.feature_id
    LEFT JOIN feature AS f4 ON source_usp_mat_id = f4.feature_id
    LEFT JOIN feature AS f5 ON candidate_usp_mat_id = f5.feature_id
    LEFT JOIN feature AS f6 ON user_pair_mat_id = f6.feature_id
    """)
    if cursor is None:
        return ValueError("Null cursor.")
    row = cursor.fetchone()
    while row is not None:
        row_md = {key: row[key] for key in row.keys()}
        assert row_md['source_usp_arr'].dtype == np.int64
        row_md['source_usp_arr'] = row_md['source_usp_arr'].astype(np.int64).reshape(-1, 2)
        row_md['candidate_usp_arr'] = row_md['candidate_usp_arr'].astype(np.int64).reshape(-1, 2)
        row_md['target_inds'] = row_md['target_inds'].astype(np.int64)
        row_md['source_usp_mat'] = row_md['source_usp_mat'].reshape(-1, config.user_feature_count)
        row_md['candidate_usp_mat'] = row_md['candidate_usp_mat'].reshape(-1, config.user_feature_count)
        row_md['user_pair_mat'] = row_md['user_pair_mat'].reshape(-1, config.user_pair_feature_count)
        assert row_md['candidate_usp_arr'].shape[0] == row_md['candidate_usp_mat'].shape[0], f"{row_md['candidate_usp_arr'].shape}.nrows != {row_md['candidate_usp_mat'].shape}.nrows"
        yield row_md
        row = cursor.fetchone()

def get_test_context_by_metadata_id(db, metadata_id, config):
    cursor = db.execute("""
    SELECT 
        metadata_id, 
        f1.feature_arr as source_usp_arr,
        f2.feature_arr as candidate_usp_arr,
        f3.feature_arr as target_inds,
        f4.feature_arr as source_usp_mat,
        f5.feature_arr as candidate_usp_mat,
        f6.feature_arr as user_pair_mat
    FROM test_context
    LEFT JOIN feature AS f1 ON source_usp_arr_id = f1.feature_id
    LEFT JOIN feature AS f2 ON candidate_usp_arr_id = f2.feature_id
    LEFT JOIN feature AS f3 ON target_inds_id = f3.feature_id
    LEFT JOIN feature AS f4 ON source_usp_mat_id = f4.feature_id
    LEFT JOIN feature AS f5 ON candidate_usp_mat_id = f5.feature_id
    LEFT JOIN feature AS f6 ON user_pair_mat_id = f6.feature_id
    WHERE metadata_id = ?
    """, (metadata_id,))
    if cursor is None:
        raise ValueError("Null cursor.")
    row = cursor.fetchone()
    if row is None:
        raise ValueError("No results for this metadata_id.")
    row_md = {key: row[key] for key in row.keys()}
    assert row_md['source_usp_arr'].dtype == np.int64
    row_md['source_usp_arr'] = row_md['source_usp_arr'].astype(np.int64).reshape(-1, 2)
    row_md['candidate_usp_arr'] = row_md['candidate_usp_arr'].astype(np.int64).reshape(-1, 2)
    row_md['target_inds'] = row_md['target_inds'].astype(np.int64)
    row_md['source_usp_mat'] = row_md['source_usp_mat'].reshape(-1, config.user_feature_count)
    row_md['candidate_usp_mat'] = row_md['candidate_usp_mat'].reshape(-1, config.user_feature_count)
    row_md['user_pair_mat'] = row_md['user_pair_mat'].reshape(-1, config.user_pair_feature_count)
    assert row_md['candidate_usp_arr'].shape[0] == row_md['candidate_usp_mat'].shape[0]
    return row_md
