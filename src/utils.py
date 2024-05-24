# src/utils.py

import logging

def setup_logging(log_file='app.log'):
    logging.basicConfig(filename=log_file, 
                        level=logging.DEBUG, 
                        format='%(asctime)s:%(levelname)s:%(message)s')

def save_model(model, model_path):
    import joblib
    joblib.dump(model, model_path)

def load_model(model_path):
    import joblib
    return joblib.load(model_path)
