"""
Run through various hyperparameters to send to Wandb for hypothesis testing and result exploration.
"""

import os
from pathlib import Path
import sys
import itertools

MODULE_PATH = Path(__file__).resolve().parents[2].resolve()
sys.path.insert(0, str(MODULE_PATH)) 

import pandas as pd
import sys
import numpy as np
from src import model_corr
from src import metric_exploration
from src.testing import df_validation
import glob
from src import utils
import matplotlib.pyplot as plt

path_combined = 'combined_dataset.csv'
path_sts = 'sts.csv'
path_qqp = 'qqp.csv'
path_qqp_sample = 'sample_qqp.csv'
path_ba = 'ba_all.txt'

METRICS = ['bleu', 
           'bleu1',
           'glove_cosine',
           'fasttext_cosine',
           'BertScore',
           'chrfScore',
           'POS Dist score',
           '1-gram_overlap',
           'ROUGE-1',
           'ROUGE-2',
           'ROUGE-l',
           'L2_score',
           'WMD']

distance_metrics = ['glove_cosine',
                    'fasttext_cosine',
                    'BertScore',
                    'POS_Dist_score',
                    'L2_score',
                    'WMD']

TOP_3_METRICS = ['WMD','BertScore', 'POS Dist score']
TOP_4_METRICS = ['WMD','BertScore', 'POS Dist score','fasttext_cosine']

filtering_heursitics = ['duration', 'random_honeypot', 'low_std', 'high_random', 'unpopular', 'sentiment_inconsistent']



def main_sweep():
    '''
    Iterating through all the permutations list below, each viable combination will be tested and sent to wandb.
    '''

    train_dataset = [path_combined]
    test_dataset = [None, path_sts,path_qqp_sample]
    # bad_annotators = [c for i in range(1,len(filtering_heursitics)+1) for c in itertools.combinations(filtering_heursitics,i)]
    bad_annotators = [None]
    # scale_features = [True]
    # rf_depth = [6]
    rf_top_n_features =  [6, None]
    metrics = [METRICS, TOP_3_METRICS, TOP_4_METRICS]

    all_options = itertools.product(train_dataset,
                                    test_dataset,
                                    bad_annotators,
                                    rf_top_n_features,
                                    metrics)


    for tr_d, tst_d, ba, rf_tn, met in all_options:

        # The function takes the same dataset when tst_d == None, no need for when they have the same dataset name
        if tr_d == tst_d:
            continue

        # ba parameter is only viable when one of the datasets is the combined dataset
        if (tr_d != path_combined) and (tst_d != path_combined) and (ba is not None):
            continue

        # The various top_n features is only something we can for when we take all the metrics
        if (len(met) != len(METRICS)) and (rf_tn is not None):
            continue

        # counter += 1

        # if counter < jump_option:
        #     continue 

        config = utils.Config(train_dataset = tr_d,
                            test_dataset = tst_d,
                            bad_annotators = ba,
                            scale_features = True,
                            scale_labels = False,
                            rf_depth = 6,
                            rf_top_n_features = rf_tn,
                            metrics=met)
        try:
            utils.wandb_logging(config, "Filtering Heuristics Sweep 2", run_wandb=True)
        except AssertionError:
            continue

def main():
    '''
    The code will log the results from the src.model_corr base score and RF based off the criteria given in the config.
    '''

    config = utils.Config(train_dataset = path_sts,
                        test_dataset = path_qqp,
                        bad_annotators = path_ba,
                        scale_features = True)

    try:
        utils.wandb_logging(config, "Semantic Similarity Sweeping Misc.")
    except AssertionError:
        print(AssertionError)

if __name__ == "__main__":
    main_sweep()


