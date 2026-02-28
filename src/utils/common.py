import os
import yaml
from box.exceptions import BoxValueError
from src.logging import logging
import pickle
import joblib

from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

@ensure_annotations
def read_yaml(path_to_yaml)->ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content=yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations    
def create_dictionaries(path_to_directories,verbose=0):
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logging.info(f"Created directory at : {path}")

@ensure_annotations
def save_bin(data, filename):
    try:
        with open(filename, 'wb') as file:
            joblib.dump(data, file)
    except Exception as e:
        raise e

@ensure_annotations
def load_bin(filename):
    try:
        with open(filename, 'rb') as file:
            return joblib.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"No such file: {filename}")
    except Exception as e:
        raise e