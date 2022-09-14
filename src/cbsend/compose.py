#!/usr/bin/env python
# Send email
# https://stackabuse.com/how-to-send-emails-with-gmail-using-python/
# https://stackoverflow.com/questions/882712/sending-html-email-using-python
# SSE = "Site Suggestion Email"
# Advice that fixed a mobile font-size issue: https://stackoverflow.com/questions/43215400/font-size-email-mobile-is-too-small

import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

import sys
from datetime import datetime, timedelta
import pytz
import argparse
import os

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

caringbridge_core_path = "/home/lana/levon003/repos/caringbridge_core"
sys.path.append(caringbridge_core_path)
import cbcore.data.paths

from . import templates


def get_site_url(site_name, participant_id, batch_id):
    return f"https://www.caringbridge.org/visit/{site_name}?utm_source=SSE&utm_medium=email&utm_campaign=SSE+email+{batch_id}&utm_content=visitsite&participant_id={participant_id}"


def get_site_journal_url(site_name, participant_id, batch_id):
    return f"https://www.caringbridge.org/visit/{site_name}/journal?utm_source=SSE&utm_medium=email&utm_campaign=SSE+email+{batch_id}&utm_content=visitjournal&participant_id={participant_id}"


def create_plaintext_rec_list(rec_list, participant_id, batch_id):
    formatted_rec_list = ""
    for r in rec_list:
        formatted_rec = f"-{r['site_title']} ({r['site_name']}) Recent journal update: \"{r['journal_title']}\" {r['journal_body']}\nRead more: {get_site_journal_url(r['site_name'], participant_id, batch_id)} \n\n"
        formatted_rec_list += formatted_rec
    return formatted_rec_list


def create_html_rec_list(rec_list, participant_id, batch_id):
    formatted_rec_list = ""
    for r in rec_list:
        if r['journal_body'].strip() == "" and len(r['journal_title'].strip()) > 2:
            formatted_rec = f'<li><a href="{get_site_url(r["site_name"], participant_id, batch_id)}" style="color: rgb(122, 110, 102);">{r["site_title"]}</a> Recent <a href="{get_site_journal_url(r["site_name"], participant_id, batch_id)}" style="color: rgb(122, 110, 102);">journal</a> update: <br>"<strong>{r["journal_title"]}</strong>" <a href="{get_site_journal_url(r["site_name"], participant_id, batch_id)}" style="color: rgb(122, 110, 102);">Read more.</a></li> \n'
        elif r['journal_body'].strip() == "":
            formatted_rec = f'<li><a href="{get_site_url(r["site_name"], participant_id, batch_id)}" style="color: rgb(122, 110, 102);">{r["site_title"]}</a> Recent <a href="{get_site_journal_url(r["site_name"], participant_id, batch_id)}" style="color: rgb(122, 110, 102);">photo</a> update: <a href="{get_site_journal_url(r["site_name"], participant_id, batch_id)}" style="color: rgb(122, 110, 102);">View.</a></li> \n'
        else:
            formatted_rec = f'<li><a href="{get_site_url(r["site_name"], participant_id, batch_id)}" style="color: rgb(122, 110, 102);">{r["site_title"]}</a> Recent <a href="{get_site_journal_url(r["site_name"], participant_id, batch_id)}" style="color: rgb(122, 110, 102);">journal</a> update: <br>"<strong>{r["journal_title"]}</strong> {r["journal_body"]}" <a href="{get_site_journal_url(r["site_name"], participant_id, batch_id)}" style="color: rgb(122, 110, 102);">Read more.</a></li> \n'
        formatted_rec_list += formatted_rec
    return formatted_rec_list


def create_email(participant_id, batch_id, to_email_address, first_name, feedback_survey_url, rec_list):
    """
    rec_list is expected to contain 1 or more recommendation dictionaries, with the following keys:
     - site_title
     - site_name
     - journal_timestamp
     - journal_title
     - journal_body
    """
    from_email_address = 'cb-suggestions@umn.edu'
    
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'CaringBridge site suggestions for you'
    msg['From'] = f"CaringBridge Suggestions Study <{from_email_address}>"
    msg['To'] = to_email_address

    html_text_template, plain_text_template = templates.batch_template_map[batch_id]
    
    formatted_rec_list = create_plaintext_rec_list(rec_list, participant_id, batch_id)
    plain_text = plain_text_template.format(
        first_name=first_name,
        formatted_rec_list=formatted_rec_list,
        feedback_survey_url=feedback_survey_url
    )
    
    formatted_rec_list = create_html_rec_list(rec_list, participant_id, batch_id)
    html_text = html_text_template.format(
        first_name=first_name,
        formatted_rec_list=formatted_rec_list,
        feedback_survey_url=feedback_survey_url
    )
    
    part1 = MIMEText(plain_text, 'plain')
    part2 = MIMEText(html_text, 'html')

    msg.attach(part1)
    msg.attach(part2)
    
    return msg
    
    
def send_email(to_email_address, msg):
    logger = logging.getLogger('cbsend.compose.send_email')
    from_email_address = 'cb-suggestions@umn.edu'
    with open('/home/lana/levon003/repos/recsys-peer-match/src/cbsend/secret.txt', 'r') as infile:
        gmail_password = infile.readlines()[0].strip()
    
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com')
        server.ehlo()
        server.login(from_email_address, gmail_password)
        server.sendmail(from_email_address, to_email_address, msg.as_string())
        server.close()
        logger.info("Message sent and connection to server closed.")
        record_email_send(to_email_address)
        return True
    except:
        logger.error("Login failed. Aborting...")
        return False
        

def record_email_send(to_email_address):
    """
    This function appends to a TSV file in the projects dir whenever an email is sent, recording who it was sent to and when.
    
    At this time, no additional info is stored about what was sent, this is just the authoritative record of how many emails were sent, to whom, and when.
    """
    timestamp = int(datetime.now().timestamp() * 1000)
    sent_rec_emails_filepath = os.path.join(cbcore.data.paths.projects_data_dir, 'recsys-peer-match', 'participant', 'sent_rec_emails.tsv')
    with open(sent_rec_emails_filepath, 'a') as outfile:
        outfile.write(f"{timestamp}\t{to_email_address}\n")
        
        
def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--raw-dir', dest='raw_dir', default='/home/zlevonian/follows/raw')
    #parser.add_argument('--text-id-txt', dest='text_id_filepath', required=True)
    #parser.add_argument('--n-processes', dest='n_processes', required=False, default=N_PROCESSES)
    args = parser.parse_args()
    
    # currently, no implemented script functionality
    # intended to be run through manual invocations of functions in this file
    
    
if __name__ == "__main__":
    main()
