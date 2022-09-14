
import cbrec.featuredb
import cbrec.paths
import cbrec.genconfig

import os
import numpy as np
from datetime import datetime

def test_feature_arrs():
    print("test_feature_arr")
    db_filepath = os.path.join(cbrec.genconfig.get_git_root_dir(), 'src/test/data/output', 'feature_test.sqlite')
    if os.path.exists(db_filepath):
        os.remove(db_filepath)
    db = cbrec.featuredb.get_db_by_filepath(db_filepath)
    try:
        cbrec.featuredb.create_db(db)
        cbrec.featuredb.insert_feature(db,
            0, 
            np.random.rand(80).astype(np.float32),
        )
        cbrec.featuredb.insert_feature(db,
            1, 
            np.random.randint(0, 1000, 80).astype(np.int64),
        )
        res_arr = cbrec.featuredb.get_feature(db, 0)
        assert res_arr.dtype == np.float32
        res_arr = cbrec.featuredb.get_feature(db, 1)
        assert res_arr.dtype == np.int64
    except Exception as ex:
        print(ex)
        raise ex
    finally:
        db.close()
    print("Finished test_feature_arr; db closed.")


def test_simple():
    print("test_simple")
    db_filepath = os.path.join(cbrec.paths.WORKING_DIR, 'sqlite', 'test.sqlite')
    os.remove(db_filepath)
    db = cbrec.featuredb.get_db_by_filepath(db_filepath)
    try:
        cbrec.featuredb.create_db(db)
        cbrec.featuredb.insert_feature(db,
            0, 
            np.random.rand(80).astype(cbrec.featuredb.NUMPY_DTYPE),
        )
        res_arr = cbrec.featuredb.get_feature(db, 0)
        print(res_arr[:10])
        print(res_arr.dtype)
        cbrec.featuredb.stream_triples(db)
    finally:
        db.close()
    print("Finished simple test; db closed.")
    
def test_json():
    print("test_json")
    db_filepath = os.path.join(cbrec.paths.WORKING_DIR, 'sqlite', 'test.sqlite')
    os.remove(db_filepath)
    db = cbrec.featuredb.get_db_by_filepath(db_filepath)
    try:
        cbrec.featuredb.create_db(db)
        cbrec.featuredb.insert_feature(db, 0, np.random.rand(80).astype(cbrec.featuredb.NUMPY_DTYPE))
        cbrec.featuredb.insert_feature(db, 1, np.random.rand(80).astype(cbrec.featuredb.NUMPY_DTYPE))
        cbrec.featuredb.insert_feature(db, 2, np.random.rand(80).astype(cbrec.featuredb.NUMPY_DTYPE))
        cbrec.featuredb.insert_triple(db,
                  0,
                  (0, 0),
                  0,
                  (1, 1),
                  1,
                  (2, 2),
                  2,
                  triple_metadata={'test_int': 5, 'test_str': 'string_value', 'test_bool': True})
        for triple in cbrec.featuredb.stream_metadata(db):
            source, target, alt, interaction_timestamp, triple_metadata = triple
            print(source, target, alt, interaction_timestamp)
            print(triple_metadata)
    finally:
        db.close()
    print("Finished JSON test; db closed.")
        
        
def test_stream_triples():
    """
    with num_triples = 1000000, feature_size = 512 + 20:
    Finished inserting features in 0:00:59.776662.
    Finished constructing triples in 0:00:04.279108.
    Retrieved first triple in 0:00:15.014457.
    Finished streaming triples in 0:00:29.148938.
    """
    print("test_stream_triples")
    db_filepath = os.path.join(cbrec.paths.WORKING_DIR, 'sqlite', 'test.sqlite')
    os.remove(db_filepath)
    db = cbrec.featuredb.get_db_by_filepath(db_filepath)
    try:
        cbrec.featuredb.create_db(db)
        num_triples = 100000
        feature_size = 512 + 20
        start = datetime.now()
        for i in range(num_triples*3):
            cbrec.featuredb.insert_feature(db,
                i, 
                np.random.rand(feature_size).astype(cbrec.featuredb.NUMPY_DTYPE),
            )
        print(f"Finished inserting features in {datetime.now() - start}.")
        
        start = datetime.now()
        for i in range(0, num_triples*3, 3):
            db.execute("""
            INSERT INTO triple (interaction_timestamp, 
            source_user_id, source_site_id, source_feature_id,
            target_user_id, target_site_id, target_feature_id,
            alt_user_id, alt_site_id, alt_feature_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (datetime.now().timestamp(), 0, 0, i, 1, 1, i+1, 2, 2, i+2))
        db.commit()
        print(f"Finished constructing triples in {datetime.now() - start}.")
        
        start = datetime.now()
        first_val = True
        for value in cbrec.featuredb.stream_triples(db):
            if first_val:
                print(f"Retrieved first triple in {datetime.now() - start}.")
                first_val = False
            interaction_timestamp, source_feature_arr, target_feature_arr, alt_feature_arr = value
            assert source_feature_arr.dtype == cbrec.featuredb.NUMPY_DTYPE
            assert target_feature_arr.dtype == cbrec.featuredb.NUMPY_DTYPE
            assert alt_feature_arr.dtype == cbrec.featuredb.NUMPY_DTYPE
        print(f"Finished streaming triples in {datetime.now() - start}.")
    finally:
        db.close()
    
if __name__ == '__main__':
    test_feature_arrs()
    #test_simple()
    #test_json()
    #test_stream_triples()