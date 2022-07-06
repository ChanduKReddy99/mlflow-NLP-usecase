import os
import argparse
import logging
from random import random
import joblib
import numpy as np
from src.utils.common_utils import read_yaml, create_dirs
from sklearn.ensemble import RandomForestClassifier

STAGE= 'STAGE-THREE'

logging.basicConfig(
    filename= os.path.join('logs', 'running_logs.log'),
    level= logging.INFO,
    format='[%(asctime)s: %(levelname)s: %(module)s]:%(message)s',
    datefmt='%m-%d %H:%M:%S',
    filemode= 'a'
)


def main(config_path, params_path):
    config= read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts= config['artifacts']

    featurized_data_dir_path= os.path.join(artifacts['ARTIFACTS_DIR'], artifacts['FEATURIZED_DATA'])
    featurized_train_data_path= os.path.join(featurized_data_dir_path, artifacts['FEATURIZED_OUT_TRAIN'])

    model_dir_path= os.path.join(artifacts['ARTIFACTS_DIR'], artifacts['MODEL_DIR'])
    create_dirs([model_dir_path])

    model_path= os.path.join(model_dir_path, artifacts['MODEL_NAME'])

    matrix= joblib.load(featurized_train_data_path)

    labels= np.squeeze(matrix[:,1].toarray())
    X= matrix[:,2:]

    logging.info(f'input matrix size: {matrix.shape}')
    logging.info(f'X matrix size: {X.shape}')
    logging.info(f'label or y matrix size: {labels.shape}')

    seed= params['train']['seed']
    n_est= params['train']['n_est']
    min_split= params['train']['min_split']

    model= RandomForestClassifier(
        n_estimators= n_est, min_samples_split= min_split, n_jobs= 2, random_state= seed
    )

    model.fit(X, labels)

    joblib.dump(model, model_path)


if __name__ == '__main__':
    parser= argparse.ArgumentParser(description='training the model')
    parser.add_argument('--config', '-c', default= 'configs/config.yaml', help= 'path to config.yaml')
    parser.add_argument('--params', '-p', default= 'params.yaml', help= 'path to params.yaml')
    parsed_args= parser.parse_args()

    try:
        logging.info('\n**************************************************')
        logging.info(f'>>>>>>>>>>> stage {STAGE} is srtated......... <<<<<<<<<<<<')
        main(config_path= parsed_args.config, params_path= parsed_args.params)
        logging.info(f'>>>>>>>>>>>> stage {STAGE} is completed successfully ... <<<<<<<<<<<<\n')

    except Exception as e:
        logging.error(e)
        logging.error(f'>>>>>>>>>>>> stage {STAGE} is failed........<<<<<<<<<<')

