
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


DUMMY_JOURNAL = "This CaringBridge site was created just recently."


class TextDatabase:
    def __init__(self, config):
        self.config = config

    def get_text_db(self):
        db = sqlite3.connect(
                self.config.raw_text_db_filepath,
                detect_types=sqlite3.PARSE_DECLTYPES
            )
        db.row_factory = sqlite3.Row
        return db


    def get_raw_journal_text(self, journal_oid):
        """
        :returns - (title, body) tuple
        """
        try:
            db = self.get_text_db()
            return self.get_raw_journal_text_from_db(db, journal_oid)
        finally:
            db.close()
            
    def get_raw_journal_text_from_db(self, db, journal_oid):
        cursor = db.execute("""SELECT title, body
                                FROM journalText
                                WHERE journal_oid = ?""", 
                            (journal_oid,))
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"Expected to find journal_oid '{journal_oid}' in text database.")
        return result['title'], result['body']
            
    def get_clean_journal_text(self, journal_oid):
        raw_title, raw_body = self.get_raw_journal_text(journal_oid)
        return clean_journal(raw_title, raw_body)
    
    
# See: https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def handle_starttag(self, tag, attrs):
        if tag == 'br':
            self.fed.append("\n")  # this adds linebreaks in place of <br> tags
    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html_text):  # this function strips HTML tags from a given text string
    s = MLStripper()
    s.feed(html_text)
    return s.get_data()


def clean_text(text):
    """
    Given a string, strips HTML tags and normalizes any wonky unicode characters.
    """
    fixed = strip_tags(text)
    fixed = ftfy.fix_text(fixed, normalization='NFKC')
    
    cleaned_text = fixed
    return cleaned_text


def clean_journal(raw_title, raw_body):
    """
    Given a title and body, cleans and combines into a single string representation.
    """
    if raw_body.startswith(DUMMY_JOURNAL):
        # auto-generated texts should be discarded
        raw_body = ""
    cleaned_text = clean_text(raw_title) + "\n" + clean_text(raw_body)
    return cleaned_text


def get_normalized_text_from_token_list(token_list):
    normalized_text = " ".join(token_list).replace(':', 'COLON').replace('|', 'PIPE').replace("\n", "NEWLINE ")
    return normalized_text


def get_lowercase_normalized_text(clean_text, lowercase=True):
    tokens = clean_text.split()
    if lowercase:
        tokens = [token.lower() for token in tokens]
    normalized_text = get_normalized_text_from_token_list(tokens)
    return normalized_text
