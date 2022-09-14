
import sqlite3
import logging

import sys
import ftfy
import json
import os
import re
from tqdm import tqdm
from html.parser import HTMLParser
import itertools


try:
    import cbrec
except:
    sys.path.append("/home/lana/levon003/repos/recsys-peer-match/src")

from cbrec import featuredb


def stream_text_features(db):
    cursor = db.execute("""
    SELECT 
        text_id, feature_arr 
    FROM textFeature
    """)
    if cursor is None:
        return ValueError("Null cursor.")
    row = cursor.fetchone()
    while row is not None:
        row_md = {key: row[key] for key in row.keys()}
        yield row_md
        row = cursor.fetchone()


def get_text_feature_arrs_from_db(db, text_id_list):
    select_command = f"SELECT text_id, feature_arr FROM textFeature WHERE text_id IN ({', '.join(['?'] * len(text_id_list))})"
    cursor = db.execute(select_command, text_id_list)
    rows = cursor.fetchall()
    
    text_arr_map = {row['text_id']: row['feature_arr'] for row in rows}
    text_feature_arrs = [text_arr_map[text_id] if text_id in text_arr_map else None for text_id in text_id_list]
    if len(text_arr_map) != len(text_id_list):
        logger = logging.getLogger("cbrec.text.embeddingdb.get_text_feature_arrs_from_db")
        #logger.warning(f"Expected to find {len(text_id_list)} text feature representations, but found only {len(text_arr_map)}.")
    return text_feature_arrs
    
    
def get_text_feature_db(config):
    db = featuredb.get_text_feature_db(config)
    return db