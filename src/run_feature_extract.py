"""
This file implements running the different semantic similiarity metrics on a dataset of paired sentences
"""

# adding cwd to path to avoid "No module named src.*" errors
import os
import sys
import nltk
sys.path.insert(0, os.path.abspath(os.getcwd()))

import pickle
import argparse
import pandas as pd
import configparser
from os import path
from src.features.bleu import Bleu
from src.features.bertscore import BertScore
from src.features.chrFScore import chrFScore
from src.features.cosine_similarites import CosineSimilarity
from src.features.elmo_euclidean_distance import EuclideanElmoDistance
from src.features.ngram_overlap import NgramOverlap
from src.features.POS_distance import POSDistance
from src.features.ROUGE import ROUGE
from src.features.WMD import WMD
from src.preprocessing import text_preprocessing


creds_path_ar = [path.join(path.dirname(os.getcwd()), "credentials.ini"), "credentials.ini"]
PATH_ROOT = ""
PATH_DATA = ""
GloVe_840B_300d_PATH = ""
Glove_twitter_27B_PATH = ""

for creds_path in creds_path_ar:
    if path.exists(creds_path):
        config_parser = configparser.ConfigParser()
        config_parser.read(creds_path)
        PATH_ROOT = config_parser['MAIN']["PATH_ROOT"]
        PATH_DATA = config_parser['MAIN']["PATH_DATA"]
        GloVe_840B_300d_PATH = config_parser['MAIN']["GloVe_840B_300d_PATH"]
        Glove_twitter_27B_PATH = config_parser['MAIN']["Glove_twitter_27B_PATH"]
        WANDB_enable = config_parser['MAIN']["WANDB_ENABLE"] == 'TRUE'
        ENV = config_parser['MAIN']["ENV"]
        break


class Config:
    """Config class for debugging in IDE"""

    def __init__(self, pckl: str, features: str = 'ALL', max_n: int = 1, exclude: str = None):
        self.pickle = pckl
        self.features = features
        self.max_n = max_n
        self.exclude = exclude


def main(args):
    nltk.download('stopwords')
    picklefile = args.pickle

    if 'pickle' in picklefile:
        with open(picklefile, 'rb') as handle:
            df = pickle.load(handle)
    else:
        df = pd.read_csv(picklefile, index_col=0)

    txt_col_format = 'text_' if 'text_1' in df.columns else 'text'

    df.dropna(subset=[f'{txt_col_format}1', f'{txt_col_format}2'], inplace=True)

    df = df[(df[f'{txt_col_format}1'].str.strip().str.split().apply(len) >= 2) &
            (df[f'{txt_col_format}2'].str.strip().str.split().apply(len) >= 2)].copy()

    df[f'{txt_col_format}1'] = text_preprocessing(df[f'{txt_col_format}1'])
    df[f'{txt_col_format}2'] = text_preprocessing(df[f'{txt_col_format}2'])

    extractors = dict(
        bleu=Bleu(txt_col_format, stopwords=True),
        cosine_similarites=CosineSimilarity(val=txt_col_format, glove_path=Glove_twitter_27B_PATH),
        elmo=EuclideanElmoDistance(val=txt_col_format),
        bert=BertScore(val=txt_col_format),
        chrf_score=chrFScore(val=txt_col_format),
        pos_distance=POSDistance(val=txt_col_format, vector_path=Glove_twitter_27B_PATH),
        wmd=WMD(val=txt_col_format, vector_path=GloVe_840B_300d_PATH),
        ngram_overlap=NgramOverlap(args.max_n, val=txt_col_format),
        rouge=ROUGE(val=txt_col_format, stopwords=True))

    features = args.features
    exclude = args.exclude
    if features == 'ALL':
        features = list(extractors.keys())
        if exclude:
            exclude = exclude.lower().split(',')
            for ex in exclude:
                features.remove(ex)
    else:
        features = features.lower().split(',')

    for feature_name, extractor in extractors.items():
        if feature_name in features:
            try:
                print(f'Running {feature_name} metric')
                df = extractor.run(df)
            except Exception as e:
                print(f'Threw error on {feature_name}')
                print(f'Threw {type(e)} with message "{e}"')

    if 'pickle' in picklefile:
        with open(picklefile, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        df.to_csv(picklefile, index=True)


###############################
#####  For debugging   ########
###############################
feats = 'bleu'
exclusion = 'wmd,elmo,bert'
datapath = '/Users/adam/SOTA/data/eval_data/qqp.csv'
arguments = Config(datapath, feats, 1, exclusion)
main(arguments)

# ################################
# # For Command-line running
# ################################
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--pickle', type=str, required=True, default='data/combined/no_annotators/combined_data_no_nans_rerun.pickle',
#                         help='pickle path for combined dataset')
#     parser.add_argument('--features', required=True, type=str, default='ALL',
#                         help='use "ALL" for all features, or comma separated list of features')
#     parser.add_argument('--exclude', required=False, type=str, default='',
#                         help='include comma separated list of features to exclude from calculation')
#     parser.add_argument('--max_n', type=int, default=1,
#                         help='maximum number of n-gram overlap score to calculate, e.g. max_n=2 creates 1-gram-overlap & 2-gram-overlap')
#
#     args = parser.parse_args()
#     main(args)
