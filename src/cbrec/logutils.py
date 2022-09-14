
import logging

def set_up_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
        
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
