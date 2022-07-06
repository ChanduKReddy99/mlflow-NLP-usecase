import os 
import logging
import argparse
from src.utils.common_utils import read_yaml, create_dirs
from src.utils.data_management import process_posts
import random


STAGE= 'STAGE-ONE'

logging.basicConfig(
    filename= os.path.join('logs', 'running_logs.log'),
    level= logging.INFO,
    format= '[%(asctime)s: %(levelname)s: %(module)s]: %(message)s',
    datefmt= '%Y-%m-%d %H:%M:%S',
    filemode= 'a'
)


def main(config_path, params_path):
    """This method get the data by reading config.yaml file and prepares the data by reading params.yaml file.

    Args:
        config_path(type:str): path to the config.yaml file
        params_path(type:str): path to the params.yaml file
    """
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    data_source= config['data_source']
    input_data= os.path.join(data_source['data_dir'], data_source['data_file'])

    split= params['prepare']['split']
    seed= params['prepare']['seed']

    random.seed(seed)

    artifacts= config['artifacts']
    prepared_data_dir_path= os.path.join(artifacts['ARTIFACTS_DIR'], artifacts['PREPARED_DATA'])

    create_dirs([prepared_data_dir_path])

    train_data_path= os.path.join(prepared_data_dir_path, artifacts['TRAIN_DATA'])
    test_data_path= os.path.join(prepared_data_dir_path, artifacts['TEST_DATA'])

   # converting .xml file to .tsv file
    encode= 'utf8'
    with open(input_data, encoding= encode) as fd_in:
        with open(train_data_path, 'w', encoding=encode) as fd_out_train:
            with open(test_data_path, 'w', encoding=encode) as fd_out_test:
                process_posts(fd_in, fd_out_train, fd_out_test, '<python>', split)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'stage01: getting data from remote resource')
    parser.add_argument('--config', '-c', type=str, default= 'configs/config.yaml', help='path to config file')
    parser.add_argument('--params', '-p', type=str, default= 'params.yaml', help= 'path to parameter file')
    parsed_args = parser.parse_args()

    try:
        logging.info('\n********************************')
        logging.info(f'>>>>>>> stage01 {STAGE} started... <<<<<<<')
        main(config_path= parsed_args.config, params_path= parsed_args.params)
        logging.info(f'>>>>>>>> stage {STAGE} completed succesfully!!! <<<<<<<<')

    except Exception as e:
        logging.error(e)
        logging.error(f'\n stage {STAGE} ailed....!')
