import os
import yaml
import logging
import pandas as pd
import json

def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content= yaml.safe_load(yaml_file)
        logging.info(f' yaml file: {path_to_yaml} loaded succesfully')
        return content

def create_dirs(path_to_dirs: list) -> dict:
    for path in path_to_dirs:
        os.makedirs(path, exist_ok=True)
        logging.info(f' created required directories at: {path}')














