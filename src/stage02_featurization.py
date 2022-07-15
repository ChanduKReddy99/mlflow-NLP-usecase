import os
import numpy as np
import argparse
import logging
from src.utils.common_utils import read_yaml, create_dirs, get_df
from src.utils.featurize import save_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import mlflow

STAGE= 'STAGE-TWO'

logging.basicConfig(
    filename= os.path.join('logs', 'running_logs.log'),
    level= logging.INFO,
    format='[%(asctime)s: %(levelname)s: %(module)s]:  %(message)s',
    datefmt='%m-%d %H:%M:%S',
    filemode= 'a'
)

def main(config_path, params_path):
    config= read_yaml(config_path)
    params= read_yaml(params_path)

    artifacts= config['artifacts']
    prepared_data_dir_path= os.path.join(artifacts['ARTIFACTS_DIR'], artifacts['PREPARED_DATA'])
    train_data_path= os.path.join(prepared_data_dir_path, artifacts['TRAIN_DATA'])
    test_data_path= os.path.join(prepared_data_dir_path, artifacts['TEST_DATA'])

    featurized_data_dir_path= os.path.join(artifacts['ARTIFACTS_DIR'], artifacts['FEATURIZED_DATA'])
    create_dirs([featurized_data_dir_path])

    featurized_train_data_path= os.path.join(featurized_data_dir_path, artifacts['FEATURIZED_OUT_TRAIN'])
    featurized_test_data_path= os.path.join(featurized_data_dir_path, artifacts['FEATURISED_OUT_TEST'])

    max_features= params['featurize']['max_features']
    ngrams= params['featurize']['ngrams']
    mlflow.log_param('max_features', max_features)
    mlflow.log_param('ngrams', ngrams)
    
    # applying word embedding to train data
    df_train= get_df(train_data_path)
    train_words= np.array(df_train.text.str.lower().values.astype('U')) #i, e U1000

    bag_of_words= CountVectorizer(
        stop_words= 'english', max_features= max_features, ngram_range=(1,ngrams)
    )

    bag_of_words.fit(train_words)
    train_words_binary_matrix= bag_of_words.transform(train_words)

    tf_idf= TfidfTransformer(smooth_idf=False)
    tf_idf.fit(train_words_binary_matrix)
    train_words_tfidf_matrix= tf_idf.transform(train_words_binary_matrix)
    save_matrix(df_train, train_words_tfidf_matrix, featurized_train_data_path)
    
    #applying word embedding to test data:
    df_test= get_df(test_data_path)
    test_words= np.array(df_test.text.str.lower().values.astype('U'))
    test_words_binary_matrix= bag_of_words.transform(test_words)
    test_words_tfidf_matrix= tf_idf.transform(test_words_binary_matrix)
    save_matrix(df_test, test_words_tfidf_matrix, featurized_test_data_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fearurization of data')
    parser.add_argument('--config','-c', default= 'configs/config.yaml', type=str, help= 'path to config.yaml')
    parser.add_argument('--params', '-p', default= 'params.yaml', type=str, help= 'path to params.yaml')
    parsed_args= parser.parse_args()

    try:
        logging.info('\n*******************************************')
        logging.info(f'>>>>>>>> stage {STAGE} started...<<<<<<<<<<')
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f'>>>>>>>>>> stage {STAGE} completed succesfully!!...<<<<<<<<')

    except Exception as e:
        logging.error(e)
        logging.error(f'>>>>>> stage {STAGE} failed....<<<<<<<<')
